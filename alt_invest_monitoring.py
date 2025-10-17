
# -*- coding: utf-8 -*-
import os, re, json, time, html, difflib
import datetime as dt
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed

import streamlit as st
import pandas as pd
import requests

st.set_page_config(page_title="Alternative Investment Monitoring", layout="wide")

with open("config.json", "r", encoding="utf-8") as f:
    CFG = json.load(f)

EXCLUDE_TITLE_KEYWORDS = CFG["EXCLUDE_TITLE_KEYWORDS"]
ALLOWED_SOURCES = set(CFG["ALLOWED_SOURCES"])
FAVORITES = CFG["favorite_categories"]
SECTOR_FILTERS = CFG.get("sector_filter_categories", {})
COMMON_FILTERS = CFG["common_filter_categories"]
SYNONYM_MAP = CFG.get("synonym_map", {})

def init_state():
    defaults = {
        "start_date": dt.date.today() - dt.timedelta(days=7),
        "end_date": dt.date.today(),
        "keyword_input": "",
        "cat_multi": [],
        "search_results": {},
        "show_limit": {},
        "remove_duplicate_articles": True,
        "require_exact_keyword_in_title_or_content": True,
        "filter_allowed_sources_only": False,
        "use_sector_filter": True,
        "sector_filter_map": {},
        "show_sentiment_badge": False,
        "enable_summary": True,
    }
    for k,v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v
init_state()

col_title, col_option1, col_option2 = st.columns([0.6, 0.2, 0.2])
with col_title:
    st.markdown(
        "<h1 style='color:#1a1a1a; margin-bottom:0.5rem;'>💼 Alternative Investment Monitoring</h1>",
        unsafe_allow_html=True
    )
with col_option1:
    st.checkbox("감성분석 배지표시", key="show_sentiment_badge")
with col_option2:
    st.checkbox("요약 기능", key="enable_summary")

st.markdown("""
<style>
[data-testid='column'] > div { gap: 0rem !important; }
.stBox { background: #fcfcfc; border-radius: 0.7em; border: 1.5px solid #e0e2e6; margin-bottom: 1.2em; padding: 1.1em; box-shadow: 0 2px 8px rgba(0,0,0,0.03); }
.news-title { word-break: break-all !important; white-space: normal !important; display:block; overflow:visible; }
</style>
""", unsafe_allow_html=True)

col_kw_input, col_kw_btn = st.columns([0.8, 0.2])
with col_kw_input:
    st.text_input(label="", value="", key="keyword_input", placeholder="쉼표(,)로 다중 키워드 입력", label_visibility="collapsed")
with col_kw_btn:
    kw_btn = st.button("검색", use_container_width=True)

st.markdown("**⭐ 섹터 선택**")
col_cat_input, col_cat_btn = st.columns([0.8, 0.2])
with col_cat_input:
    selected_categories = st.multiselect(
        label="",
        options=list(FAVORITES.keys()),
        key="cat_multi",
        label_visibility="collapsed"
    )
with col_cat_btn:
    cat_btn = st.button("🔍 검색", use_container_width=True)

date_col1, date_col2 = st.columns([1,1])
with date_col1:
    st.date_input("시작일", key="start_date")
with date_col2:
    st.date_input("종료일", key="end_date")

with st.expander("🧩 공통 필터 옵션 (항상 적용됨)"):
    for major, subs in COMMON_FILTERS.items():
        st.markdown(f"**{major}**: {', '.join(subs)}" )

with st.expander("📊 섹터별 필터 옵션 (대분류별 세부 이슈 필터링)"):
    st.checkbox("이 필터 적용", key="use_sector_filter")
    updated_map = {}
    for sector in selected_categories:
        options = SECTOR_FILTERS.get(sector, [])
        default_selected = st.session_state.get("sector_filter_map", {}).get(sector, options)
        selected_sub = st.multiselect(
            f"{sector} 세부 이슈 키워드",
            options=options,
            default=default_selected,
            key=f"subfilter_sector_{sector}"
        )
        updated_map[sector] = selected_sub
    st.session_state["sector_filter_map"] = updated_map

with st.expander("🔍 키워드 필터 옵션"):
    st.checkbox("키워드가 제목 또는 본문에 포함된 기사만 보기", key="require_exact_keyword_in_title_or_content")
    st.checkbox("중복 기사 제거", key="remove_duplicate_articles")
    st.checkbox("특정 언론사만 검색", key="filter_allowed_sources_only", help="ALLOWED_SOURCES 목록만 허용")


def expand_keywords_with_synonyms(keywords):
    out = []
    for kw in keywords:
        out.append(kw)
        out.extend(SYNONYM_MAP.get(kw, []))
    seen=set(); res=[]
    for k in out:
        if k not in seen:
            res.append(k); seen.add(k)
    return res

def exclude_by_title_keywords(title):
    for w in EXCLUDE_TITLE_KEYWORDS:
        if w in title:
            return True
    return False

def infer_source_from_url(url):
    domain = urlparse(url).netloc
    if domain.startswith("www."):
        domain = domain[4:]
    return domain

NAVER_CLIENT_ID = os.environ.get("NAVER_CLIENT_ID")
NAVER_CLIENT_SECRET = os.environ.get("NAVER_CLIENT_SECRET")


def fetch_naver_news(query, start_date=None, end_date=None, limit=300, require_keyword_in_title=False):
    if not NAVER_CLIENT_ID or not NAVER_CLIENT_SECRET:
        return []
    headers = {"X-Naver-Client-Id": NAVER_CLIENT_ID, "X-Naver-Client-Secret": NAVER_CLIENT_SECRET}
    articles=[]
    for start in range(1, 1001, 100):
        if len(articles) >= limit: break
        params={"query": query, "display":100, "start":start, "sort":"date"}
        try:
            r=requests.get("https://openapi.naver.com/v1/search/news.json", headers=headers, params=params, timeout=20)
        except Exception:
            break
        if r.status_code != 200: break
        items = r.json().get("items", [])
        for it in items:
            title = html.unescape(re.sub("<.*?>","",it["title"]))
            desc  = html.unescape(re.sub("<.*?>","",it["description"]))
            try:
                pub_date = dt.datetime.strptime(it["pubDate"], "%a, %d %b %Y %H:%M:%S %z").date()
            except:
                continue
            if start_date and pub_date < start_date: continue
            if end_date and pub_date > end_date: continue
            if require_keyword_in_title and query.lower() not in title.lower(): continue
            if exclude_by_title_keywords(title): continue
            link = it.get("originallink") or it.get("link")
            source = infer_source_from_url(link) if link else "Naver"
            articles.append({
                "title": title, "description": desc, "link": link,
                "date": pub_date.strftime("%Y-%m-%d"), "source": source, "검색어": query
            })
        if len(items) < 100: break
    return articles[:limit]

def is_similar(a, b, th=0.5):
    return difflib.SequenceMatcher(None, a, b).ratio() >= th

def remove_duplicates(arts):
    uniq=[]; titles=[]
    for a in arts:
        t=a.get("title","" )
        if all(not is_similar(t, tt) for tt in titles):
            uniq.append(a); titles.append(t)
    return uniq

def get_alt_invest_credit_keywords():
    return CFG.get("sector_filter_categories", {})

from openai import OpenAI
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

def extract_article_text(url, fallback_desc=None):
    try:
        import newspaper
        art = newspaper.Article(url, language='ko')
        art.download(); art.parse()
        text = art.text.strip()
        if len(text) < 80:
            raise Exception("short")
        return text
    except Exception:
        return fallback_desc or ""

def summarize_article(url, title, description, target_keyword):
    if not client:
        return ("OpenAI API 키 미설정", "", "감성 추출 실패", "", "", description or "")
    full_text = extract_article_text(url, fallback_desc=description) or (title + "\n" + (description or ""))
    sector_kw_map = get_alt_invest_credit_keywords()
    prompt = f"""
[섹터별 대체투자 핵심 이슈 키워드]
{{json.dumps(sector_kw_map, ensure_ascii=False, indent=2)}}

아래 기사 본문을 분석해, '{target_keyword}'와 직접 관련된 **대체투자 관점**에서만 응답하시오.

1. [심층 시사점]: 신용평가 리포트 톤으로 작성. 
   - (거래/조달) 리파이낸싱·만기벽·Covenant(headroom/트리거)·레버리지 변화·유동성 지원 여부
   - (운영/현금흐름) DSCR·가동률/이용률·가격(PPA/SMP/임대료)·CapEx/O&M 변화·비용 민감도
   - (계약/규제) 장기계약 안정성(Anchor·LTA), 컨세션/정책/규제 변화의 등급 영향
   - (방향성) 등급/전망에 대한 상향·하향 요인과 모니터링 포인트를 2~3문장 이상으로
2. [한 줄 시사점]: 위 내용을 **투자자 의사결정** 관점에서 한 문장으로.
3. [한 줄 요약]: 주체·핵심 사건·결과를 간결히.
4. [검색 키워드]: 기사에 사용된 관련 키워드(콤마 구분).
5. [감성]: 긍정/부정 중 하나(신용도 영향 기준).
6. [주요 키워드]: 인물·기업·조직명만 콤마로.

[기사 본문]
{full_text}
""".strip()

    try:
        res = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role":"system","content":"너는 대체투자·신용평가 분석가다. 정확·간결하게 답하라."},
                      {"role":"user","content":prompt}],
            temperature=0, max_tokens=800
        )
        answer = res.choices[0].message.content.strip()
        def pick(tag):
            m = re.search(rf"\[{tag}\]:\s*([\s\S]+?)(?=\n\[\w+ ?\w*\]:|$)", answer)
            return m.group(1).strip() if m else ""
        senti_raw = pick("감성")
        senti = "긍정" if senti_raw in ("긍정","positive","Positive") else ("부정" if senti_raw in ("부정","negative","Negative") else "감성 추출 실패")
        return (
            pick("한 줄 요약") or "요약 실패",
            pick("검색 키워드"),
            senti,
            pick("심층 시사점") or "시사점 추출 실패",
            pick("한 줄 시사점") or "",
            full_text
        )
    except Exception as e:
        return (f"요약 오류: {e}", "", "감성 추출 실패", "", "", full_text)

def run_search_by_keywords(keywords):
    expanded = expand_keywords_with_synonyms(keywords)
    results = []
    with ThreadPoolExecutor(max_workers=min(5, len(expanded) or 1)) as ex:
        futs = {ex.submit(fetch_naver_news, kw, st.session_state["start_date"], st.session_state["end_date"],
                          require_keyword_in_title=st.session_state["require_exact_keyword_in_title_or_content"]): kw for kw in expanded}
        for fu in as_completed(futs):
            kw = futs[fu]
            try:
                arts = fu.result()
                for a in arts:
                    a["키워드"] = kw
                results.extend(arts)
            except Exception:
                pass
    if st.session_state["remove_duplicate_articles"]:
        results = remove_duplicates(results)
    for main in keywords:
        st.session_state["search_results"][main] = [a for a in results if a["검색어"] == main or a["키워드"] == main]

def run_search_by_sectors(sectors):
    entities = []
    for s in sectors:
        entities += FAVORITES.get(s, [])
    if not entities: return
    run_search_by_keywords(list(dict.fromkeys(entities)))

keywords_input = [k.strip() for k in st.session_state["keyword_input"].split(",") if k.strip()]
if kw_btn and keywords_input:
    st.session_state["search_results"] = {}
    run_search_by_keywords(keywords_input)

if cat_btn and selected_categories:
    st.session_state["search_results"] = {}
    run_search_by_sectors(selected_categories)

ALL_COMMON = [k for v in COMMON_FILTERS.values() for k in v]

def text_contains_any(text, keywords):
    t = (text or "").lower()
    return any(k.lower() in t for k in keywords)

def article_passes_all_filters(article):
    if exclude_by_title_keywords(article.get("title","")): return False
    try:
        d = dt.datetime.strptime(article.get("date",""), "%Y-%m-%d").date()
        if d < st.session_state["start_date"] or d > st.session_state["end_date"]:
            return False
    except: 
        return False
    if st.session_state["filter_allowed_sources_only"]:
        src = (article.get("source","") or "").lower()
        if src.startswith("www."): src = src[4:]
        if src not in ALLOWED_SOURCES:
            return False
    if not text_contains_any((article.get("title","") + " " + article.get("description","")), ALL_COMMON):
        return False
    sector_passed = True
    if st.session_state["use_sector_filter"]:
        key = article.get("키워드") or article.get("검색어")
        matched_sector=None
        for sec, ents in FAVORITES.items():
            if key in ents:
                matched_sector = sec; break
        if matched_sector:
            sub_filters = st.session_state.get("sector_filter_map", {}).get(matched_sector, [])
            if sub_filters:
                sector_passed = text_contains_any((article.get("title","") + " " + article.get("description","")), sub_filters)
    kw_pass = True
    if st.session_state["require_exact_keyword_in_title_or_content"]:
        keys=[]
        if keywords_input: keys += keywords_input
        for s in selected_categories: keys += FAVORITES.get(s, [])
        kw_pass = text_contains_any((article.get("title","")+ " " + article.get("description","")), keys)
    if not (sector_passed or kw_pass):
        return False
    return True

if st.session_state["search_results"]:
    st.markdown("### 🔎 검색 결과")
    for bucket, articles in st.session_state["search_results"].items():
        st.markdown(f"#### • {bucket}")
        cnt = 0
        for a in articles:
            if not article_passes_all_filters(a): 
                continue
            cnt += 1
            with st.container():
                st.markdown(f"**<span class='news-title'>{a['title']}</span>**", unsafe_allow_html=True)
                st.caption(f"{a.get('date')} · {a.get('source','')} · 키워드: {a.get('키워드') or a.get('검색어')}")
                st.write(a.get("description",""))
                if a.get("link"): st.markdown(f"[원문보기]({a.get('link')})")
                if st.session_state["enable_summary"]:
                    with st.spinner("요약 중..."):
                        one_line, kw, senti, implication, short_imp, _ = summarize_article(a.get("link"), a.get("title"), a.get("description"), a.get("키워드") or a.get("검색어"))
                    st.markdown(f"- **한 줄 요약**: {one_line}")
                    st.markdown(f"- **한 줄 시사점**: {short_imp}")
                    st.markdown(f"- **심층 시사점**: {implication}")
                    st.markdown(f"- **감성**: {senti}")
        if cnt == 0:
            st.info("표시할 기사가 없습니다. (필터에 모두 걸러짐)")
else:
    st.info("뉴스 검색 결과가 없습니다. 먼저 검색을 실행해 주세요.")
