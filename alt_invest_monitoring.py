#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Streamlit-only Alternative Investment Monitoring (News → Telegram)

- config.json의 ALT_INVEST 섹션을 읽어 대체투자 관련 뉴스를 수집/필터/정렬
- 미리보기 및 텔레그램 전송 UI 제공
- 검색 인프라: Naver News OpenAPI, NewsAPI (환경변수 필요)
- LLM(선택): OpenAI Chat Completions API (환경변수 필요)

실행:
    streamlit run alt_invest_monitoring.py

필요 환경변수:
    NAVER_CLIENT_ID, NAVER_CLIENT_SECRET   # Naver News API (선택 권장)
    NEWSAPI_KEY                             # NewsAPI (선택 권장)
    OPENAI_API_KEY                          # LLM 사용시
    TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID    # 텔레그램 전송시
선택 환경변수:
    ALT_CFG                                 # config.json 경로(기본: ./config.json)
    ALT_SENT_CACHE                          # 중복전송 방지 캐시 파일 경로(기본: ./sent_cache_alt.json)
"""

import os
import re
import json
import time
import logging
import datetime as dt
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from urllib.parse import urlparse

import requests
import streamlit as st

# --------- OpenAI (optional) ---------
try:
    from openai import OpenAI
    _OPENAI_AVAILABLE = True
except Exception:
    _OPENAI_AVAILABLE = False

# --------- Globals ---------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("alt_invest_monitor")

APP_TZ = dt.timezone(dt.timedelta(hours=9))  # Asia/Seoul
CACHE_PATH = os.getenv("ALT_SENT_CACHE", "sent_cache_alt.json")
DEFAULT_CONFIG_PATH = os.getenv("ALT_CFG", "config.json")


# =========================
# Utilities
# =========================
def now_kst() -> dt.datetime:
    return dt.datetime.now(tz=APP_TZ)

def parse_date(ts: str) -> Optional[dt.datetime]:
    if not ts:
        return None
    try:
        return dt.datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone(APP_TZ)
    except Exception:
        pass
    try:
        from email.utils import parsedate_to_datetime
        return parsedate_to_datetime(ts).astimezone(APP_TZ)
    except Exception:
        return None

def sha1(s: str) -> str:
    import hashlib
    return hashlib.sha1((s or "").encode("utf-8")).hexdigest()

def normalize_title(t: str) -> str:
    x = re.sub(r"\s+", " ", t or "").strip().lower()
    x = re.sub(r"\[[^\]]+\]", "", x)   # [속보], [단독] 제거
    x = re.sub(r"\([^)]*\)", "", x)    # (영상) 등 제거
    x = re.sub(r"[-–—:|·•]+", " ", x)
    return x.strip()

def fuzzy_ratio(a: str, b: str) -> float:
    import difflib
    return difflib.SequenceMatcher(None, a or "", b or "").ratio()

def load_json(path: str, default):
    if not os.path.exists(path):
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default

def save_json(path: str, obj):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


# =========================
# Config dataclasses
# =========================
@dataclass
class LLMConfig:
    enable: bool = False
    model: str = "gpt-4o-mini"
    temperature: float = 0.2
    top_p: float = 1.0
    system_prompt: str = (
        "너는 보험사 자산운용 대체투자 심사 담당자다.\n"
        "기사의 본문/제목/요약을 바탕으로 '대체투자(부동산/인프라/에너지/물류/데이터센터/해운/항공기/인수금융/PE 등)'\n"
        "관련성, 신용위험·거래현황·규제·시장동향 등 실무 중요도를 0~1로 평가하고,\n"
        "핵심 근거 키워드 3~6개를 추출해라. JSON만 반환해라.\n"
        '{"score": 0.00, "reason": "...", "tags": ["...","..."]}'
    )
    threshold: float = 0.68

@dataclass
class AltConfig:
    lookback_hours: int = 24
    max_per_keyword: int = 10
    allow_domains: Optional[List[str]] = None
    deny_domains: Optional[List[str]] = None
    include_keywords: Optional[List[str]] = None
    exclude_keywords: Optional[List[str]] = None
    keywords: Optional[Dict[str, List[str]]] = None
    llm: LLMConfig = field(default_factory=LLMConfig)  # ← mutable default 방지
    show_domain_in_telegram: bool = False
    send_top_n: int = 0

    def merge_defaults(self, base: Dict[str, Any]):
        for k in ["allow_domains", "deny_domains", "include_keywords", "exclude_keywords"]:
            if getattr(self, k) is None:
                setattr(self, k, base.get(k, []))
        if self.keywords is None:
            self.keywords = {}


# =========================
# Data model
# =========================
@dataclass
class NewsItem:
    title: str
    url: str
    source: str
    published_at: Optional[str]
    summary: Optional[str] = None
    keyword_bucket: Optional[str] = None
    llm_score: Optional[float] = None
    llm_tags: Optional[List[str]] = None


# =========================
# Providers
# =========================
def naver_news_search(query: str, from_ts: dt.datetime, to_ts: dt.datetime, size=30) -> List[NewsItem]:
    cid = os.getenv("NAVER_CLIENT_ID", "")
    csecret = os.getenv("NAVER_CLIENT_SECRET", "")
    if not (cid and csecret):
        return []
    url = "https://openapi.naver.com/v1/search/news.json"
    headers = {"X-Naver-Client-Id": cid, "X-Naver-Client-Secret": csecret}
    params = {"query": query, "display": min(100, size), "start": 1, "sort": "date"}
    r = requests.get(url, headers=headers, params=params, timeout=15)
    r.raise_for_status()
    data = r.json().get("items", [])
    out: List[NewsItem] = []
    for it in data:
        link = it.get("link") or it.get("originallink")
        title = re.sub("<[^>]+>", "", it.get("title", ""))
        pub = it.get("pubDate")
        out.append(NewsItem(title=title, url=link, source=urlparse(link).netloc if link else "naver", published_at=pub))
    return out

def newsapi_search(query: str, from_ts: dt.datetime, to_ts: dt.datetime, size=50) -> List[NewsItem]:
    key = os.getenv("NEWSAPI_KEY", "")
    if not key:
        return []
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": query,
        "from": (from_ts - dt.timedelta(hours=1)).isoformat(),
        "to": (to_ts + dt.timedelta(hours=1)).isoformat(),
        "pageSize": min(100, size),
        "language": "ko",
        "sortBy": "publishedAt",
    }
    headers = {"Authorization": f"Bearer {key}"}
    r = requests.get(url, headers=headers, params=params, timeout=20)
    r.raise_for_status()
    data = r.json().get("articles", [])
    out: List[NewsItem] = []
    for a in data:
        url_ = a.get("url")
        out.append(NewsItem(
            title=a.get("title") or "",
            url=url_,
            source=urlparse(url_).netloc if url_ else (a.get("source") or {}).get("name", ""),
            published_at=a.get("publishedAt"),
        ))
    return out


# =========================
# Collection / Filtering
# =========================
def collect_alt(cfg: AltConfig) -> List[NewsItem]:
    to_ts = now_kst()
    from_ts = to_ts - dt.timedelta(hours=cfg.lookback_hours)
    results: List[NewsItem] = []
    for bucket, keys in (cfg.keywords or {}).items():
        if not keys:
            continue
        q = " OR ".join([f'"{k}"' if " " in k else k for k in keys])
        items: List[NewsItem] = []
        items += naver_news_search(q, from_ts, to_ts, size=50)
        items += newsapi_search(q, from_ts, to_ts, size=50)
        for it in items:
            it.keyword_bucket = bucket
        results.extend(items)
    return results

def domain_allowed(host: str, allow: List[str], deny: List[str]) -> bool:
    if deny and any(d for d in deny if d and d in host):
        return False
    if allow:
        return any(a for a in allow if a and a in host)
    return True

def keyword_filter_pass(title: str, include_words: List[str], exclude_words: List[str]) -> bool:
    t = title or ""
    if exclude_words and any((w and w in t) for w in exclude_words):
        return False
    if include_words:
        return any((w and w in t) for w in include_words)
    return True

def deduplicate(items: List[NewsItem]) -> List[NewsItem]:
    seen_url = set()
    seen_hash: Dict[str, str] = {}
    out: List[NewsItem] = []
    for it in items:
        if not it.url:
            continue
        p = urlparse(it.url)
        ukey = f"{p.scheme}://{p.netloc}{p.path}"
        if ukey in seen_url:
            continue
        seen_url.add(ukey)

        nt = normalize_title(it.title)
        h = sha1(nt)
        if h in seen_hash:
            continue
        # fuzzy near-dup
        skip = False
        for h2, t2 in list(seen_hash.items()):
            if abs(len(nt) - len(t2)) <= 10 and fuzzy_ratio(nt, t2) >= 0.86:
                skip = True; break
        if skip:
            continue
        seen_hash[h] = nt
        out.append(it)
    return out

def apply_llm(items: List[NewsItem], cfg: AltConfig) -> List[NewsItem]:
    if not (cfg.llm.enable and _OPENAI_AVAILABLE and os.getenv("OPENAI_API_KEY")):
        return items
    client = OpenAI()
    kept: List[NewsItem] = []
    for it in items:
        content = f"제목: {it.title}\nURL: {it.url}\n출처: {it.source}\n"
        try:
            resp = client.chat.completions.create(
                model=cfg.llm.model,
                temperature=cfg.llm.temperature,
                top_p=cfg.llm.top_p,
                messages=[
                    {"role": "system", "content": cfg.llm.system_prompt},
                    {"role": "user", "content": content},
                ]
            )
            txt = resp.choices[0].message.content.strip()
            data = json.loads(txt)
            it.llm_score = float(data.get("score", 0.0))
            it.llm_tags = data.get("tags", [])
            if it.llm_score is None:
                it.llm_score = 0.0
        except Exception as e:
            log.warning("LLM parse failed: %s", e)
            it.llm_score = 0.0
        if (it.llm_score or 0.0) >= cfg.llm.threshold:
            kept.append(it)
    return kept

def filter_and_rank(items: List[NewsItem], cfg: AltConfig, base_defaults: Dict[str, Any]) -> List[NewsItem]:
    cfg.merge_defaults(base_defaults)

    tmp: List[NewsItem] = []
    for it in items:
        host = urlparse(it.url).netloc if it.url else ""
        if not domain_allowed(host, cfg.allow_domains or [], cfg.deny_domains or []):
            continue
        if not keyword_filter_pass(it.title, cfg.include_keywords or [], cfg.exclude_keywords or []):
            continue
        tmp.append(it)

    tmp = deduplicate(tmp)
    tmp = apply_llm(tmp, cfg)

    def sort_key(x: NewsItem):
        t = parse_date(x.published_at) or now_kst()
        sc = x.llm_score if (x.llm_score is not None) else 0.0
        return (-sc, -t.timestamp())

    tmp.sort(key=sort_key)
    if cfg.send_top_n and cfg.send_top_n > 0:
        tmp = tmp[: cfg.send_top_n]
    return tmp


# =========================
# Telegram
# =========================
def tg_send_message(text: str) -> bool:
    token = os.getenv("TELEGRAM_BOT_TOKEN", "")
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "")
    if not (token and chat_id):
        log.error("TELEGRAM env not set")
        return False
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    r = requests.post(url, json={
        "chat_id": chat_id,
        "text": text,
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
    }, timeout=20)
    try:
        r.raise_for_status()
        return True
    except Exception as e:
        log.error("Telegram send failed: %s | %s", e, r.text[:200])
        return False

def render_telegram(items: List[NewsItem], cfg: AltConfig) -> str:
    if not items:
        return "📌 대체투자 모니터링: 신규 적합 기사 없음"
    lines = ["📌 대체투자 모니터링 Top"]
    for it in items:
        host = urlparse(it.url).netloc if it.url else ""
        t = parse_date(it.published_at)
        ts = t.strftime("%Y-%m-%d %H:%M") if t else ""
        tag_txt = f' | {" ".join("#"+x for x in (it.llm_tags or [])[:5])}' if it.llm_tags else ""
        dom_txt = f" — {host}" if cfg.show_domain_in_telegram and host else ""
        score_txt = f" (LLM:{it.llm_score:.2f})" if it.llm_score is not None else ""
        kb = f"[{it.keyword_bucket}] " if it.keyword_bucket else ""
        lines.append(f"• {kb}{it.title}{score_txt}{tag_txt}\n  {it.url}{dom_txt} ({ts})")
    return "\n".join(lines)


# =========================
# Cache (중복 전송 방지)
# =========================
def load_sent_cache() -> Dict[str, float]:
    return load_json(CACHE_PATH, {})

def mark_sent(items: List[NewsItem]):
    cache = load_sent_cache()
    now = time.time()
    for it in items:
        cache[sha1(it.url)] = now
    save_json(CACHE_PATH, cache)

def drop_already_sent(items: List[NewsItem]) -> List[NewsItem]:
    cache = load_sent_cache()
    out = []
    for it in items:
        if sha1(it.url) in cache:
            continue
        out.append(it)
    return out


# =========================
# Streamlit App
# =========================
def run_app():
    st.set_page_config(page_title="대체투자 모니터링", layout="wide")
    st.title("📡 대체투자 모니터링 (News → Telegram)")

    cfg_path = st.sidebar.text_input("config.json 경로", value=DEFAULT_CONFIG_PATH)
    raw = load_json(cfg_path, {})
    base_defaults = raw.get("DEFAULTS", {})
    alt_raw = raw.get("ALT_INVEST", {})

    if not alt_raw:
        st.warning("config.json ▶ ALT_INVEST 섹션이 비어있습니다. 키워드를 설정하세요.")
    if not os.getenv("NAVER_CLIENT_ID") or not os.getenv("NAVER_CLIENT_SECRET"):
        st.info("Naver News API 환경변수가 없으면 결과가 제한될 수 있습니다.")
    if not os.getenv("NEWSAPI_KEY"):
        st.info("NEWSAPI_KEY가 없으면 결과가 제한될 수 있습니다.")

    lookback = st.sidebar.number_input("룩백(시간)", 1, 168, value=int(alt_raw.get("lookback_hours", 24)))
    threshold = st.sidebar.slider("LLM Threshold", 0.0, 1.0, float((alt_raw.get("llm") or {}).get("threshold", 0.68)), 0.01)
    send_top_n = st.sidebar.number_input("전송 최대 N(0=제한없음)", 0, 100, value=int(alt_raw.get("send_top_n", 0)))
    show_domain = st.sidebar.checkbox("도메인 표시", value=bool(alt_raw.get("show_domain_in_telegram", False)))
    use_llm = st.sidebar.checkbox("LLM 필터 사용", value=bool((alt_raw.get("llm") or {}).get("enable", True)))

    cfg = AltConfig(
        lookback_hours=int(lookback),
        max_per_keyword=int(alt_raw.get("max_per_keyword", 10)),
        allow_domains=alt_raw.get("allow_domains"),
        deny_domains=alt_raw.get("deny_domains"),
        include_keywords=alt_raw.get("include_keywords"),
        exclude_keywords=alt_raw.get("exclude_keywords"),
        keywords=alt_raw.get("keywords"),
        llm=LLMConfig(
            enable=use_llm,
            model=(alt_raw.get("llm") or {}).get("model", "gpt-4o-mini"),
            temperature=float((alt_raw.get("llm") or {}).get("temperature", 0.2)),
            top_p=float((alt_raw.get("llm") or {}).get("top_p", 1.0)),
            system_prompt=(alt_raw.get("llm") or {}).get("system_prompt", LLMConfig().system_prompt),
            threshold=float(threshold),
        ),
        show_domain_in_telegram=bool(show_domain),
        send_top_n=int(send_top_n),
    )

    st.sidebar.caption("환경변수: NAVER_CLIENT_ID / NAVER_CLIENT_SECRET / NEWSAPI_KEY / OPENAI_API_KEY / TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔎 수집 · 미리보기"):
            try:
                items = collect_alt(cfg)
                items = filter_and_rank(items, cfg, base_defaults)
                preview_txt = render_telegram(items, cfg)
                st.text_area("Telegram 미리보기", value=preview_txt, height=420)
                if items:
                    st.success(f"수집/필터링 완료: {len(items)}건")
                else:
                    st.info("적합 기사 없음")
            except Exception as e:
                st.error(f"수집/필터링 중 오류: {e}")

    with col2:
        if st.button("📨 텔레그램 전송"):
            try:
                items = collect_alt(cfg)
                items = filter_and_rank(items, cfg, base_defaults)
                items = drop_already_sent(items)
                txt = render_telegram(items, cfg)
                ok = tg_send_message(txt) if items else tg_send_message("📌 대체투자 모니터링: 신규 적합 기사 없음")
                if ok and items:
                    mark_sent(items)
                st.success("전송 완료" if ok else "전송 실패")
                st.text_area("전송 본문", value=txt, height=420)
            except Exception as e:
                st.error(f"전송 중 오류: {e}")

    st.markdown("---")
    st.caption("※ 키워드는 config.json ▶ ALT_INVEST.keywords 에서 관리합니다.")

# run
if __name__ == "__main__":
    run_app()
