import os
import streamlit as st
import pandas as pd
from io import BytesIO
import requests
import re
from datetime import datetime, timedelta
import telepot
from openai import OpenAI
import newspaper
import difflib
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import html
import json
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
import pandas as pd

# --- config.json 로드 ---
with open("config.json", "r", encoding="utf-8") as f:
    config = json.load(f)

EXCLUDE_TITLE_KEYWORDS = config["EXCLUDE_TITLE_KEYWORDS"] # --- 제외 키워드 ---
ALLOWED_SOURCES = set(config["ALLOWED_SOURCES"]) # 필터링할 언론사 도메인 리스트 (www. 제거된 도메인 기준)
favorite_categories = config["favorite_categories"] # --- 즐겨찾기 카테고리(변경 금지) ---
excel_company_categories = config["excel_company_categories"]
common_filter_categories = config["common_filter_categories"] # --- 공통 필터 옵션(대분류/소분류 없이 모두 적용) ---
industry_filter_categories = config["industry_filter_categories"] # --- 산업별 필터 옵션 ---
SYNONYM_MAP = config["synonym_map"]
kiscd_map = config.get("kiscd_map", {})
kr_compcd_map = config.get("kr_COMP_CD_map", {})

# 공통 필터 키워드 전체 리스트 생성
ALL_COMMON_FILTER_KEYWORDS = []
for keywords in common_filter_categories.values():
    ALL_COMMON_FILTER_KEYWORDS.extend(keywords)

def make_key(*parts):
    return "_".join(re.sub(r"\W+", "", str(p)) for p in parts).lower()

def extract_file_url(js_href: str) -> str:
    if not js_href or not js_href.startswith("javascript:fn_file"):
        return ""
    m = re.search(r"fn_file\((.*)\)", js_href)
    if not m:
        return ""
    args_str = m.group(1)
    args = [arg.strip().strip("'\"") for arg in args_str.split(",")]
    if len(args) < 4:
        return ""
    file_name = args[3]
    return f"https://www.kisrating.com/common/download.do?filename={file_name}"

def extract_reports_and_research(html: str) -> dict:
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html, 'html.parser')
    result = {
        "평가리포트": [],
        "관련리서치": [],
        "신용등급상세": []
    }

    # 평가리포트, 관련리서치 테이블 로직 그대로
    tables = soup.select('div.table_ty1 > table')
    for table in tables:
        caption = table.find('caption')
        if not caption:
            continue
        caption_text = caption.text.strip()

        if caption_text == "평가리포트":
            rows = table.select('tbody > tr')
            for tr in rows:
                tds = tr.find_all('td')
                if len(tds) < 4:
                    continue
                report_type = tds[0].text.strip()
                a_tag = tds[1].find('a')
                title = a_tag.text.strip() if a_tag else ''
                date = tds[2].text.strip()
                eval_type = tds[3].text.strip()
                result["평가리포트"].append({
                    "종류": report_type,
                    "리포트": title,
                    "일자": date,
                    "평가종류": eval_type
                })
        elif caption_text == "관련 리서치":
            rows = table.select('tbody > tr')
            for tr in rows:
                tds = tr.find_all('td')
                if len(tds) < 4:
                    continue
                category = tds[0].text.strip()
                a_tag = tds[1].find('a')
                title = a_tag.text.strip() if a_tag else ''
                date = tds[2].text.strip()
                result["관련리서치"].append({
                    "구분": category,
                    "제목": title,
                    "일자": date
                })

    # 신용등급상세 추가 (ex. 현대해상 등급 테이블)
    # 기존 extract_credit_details 코드를 활용하여 리스트를 추가
    result["신용등급상세"] = extract_credit_details(html)

    return result

# 별도 함수로 신용등급상세 추출
def extract_credit_details(html):
    soup = BeautifulSoup(html, 'html.parser')
    results = []
    items = soup.select('div.list li')
    for item in items:
        key_tag = item.find('dt') or item.find('strong')
        kind = key_tag.get_text(strip=True) if key_tag else None
        if not kind:
            continue
        # 등급
        grade_tag = item.find('span', string='등급')
        grade_val = ""
        if grade_tag:
            grade_node = grade_tag.find_next(['a', 'strong'])
            grade_val = grade_node.get_text(strip=True) if grade_node else ""
        # Outlook/Watchlist
        outlook_tag = item.find('span', string=lambda s: s and ('Outlook' in s or 'Watchlist' in s))
        outlook_val = outlook_tag.next_sibling.strip() if outlook_tag and outlook_tag.next_sibling else ""
        # 평가일
        eval_date_tag = item.find('span', string='평가일')
        eval_date_val = eval_date_tag.next_sibling.strip() if eval_date_tag and eval_date_tag.next_sibling else ""
        # 평가의견
        eval_opinion_tag = item.find('span', string='평가의견')
        eval_opinion_val = ""
        if eval_opinion_tag:
            next_node = eval_opinion_tag.find_next('a')
            if next_node:
                eval_opinion_val = next_node.get_text(strip=True)
            else:
                eval_opinion_val = eval_opinion_tag.find_next(string=True).strip()
        results.append({
            "종류": kind,
            "등급": grade_val,
            "Outlook/Watchlist": outlook_val,
            "평가일": eval_date_val,
            "평가의견": eval_opinion_val
        })
    return results

def fetch_and_display_reports(companies_map):
    import pandas as pd
    import requests
    import time
    from bs4 import BeautifulSoup

    def extract_table_after_marker(soup, marker_str):
        marker = None
        for tag in soup.find_all(['b', 'strong', 'h2', 'h3', 'span']):
            if marker_str in tag.get_text():
                marker = tag
                break
        return marker.find_next('table') if marker else None

    def parse_grade_table_html(table_tag):
        try:
            dfs = pd.read_html(str(table_tag), header=[0, 1])
            df = dfs[0]
            df.columns = [
                '_'.join([str(l) for l in col if str(l) not in ['nan', 'None']]).strip()
                for col in df.columns.values
            ]
            if all(('Unnamed' in col or col == '' or col.lower() == 'none') for col in df.columns):
                raise Exception("헤더 파싱 실패 - 단일라인 헤더 시도")
            return df
        except Exception:
            try:
                dfs = pd.read_html(str(table_tag), header=0)
                df = dfs[0]
                df.columns = [str(col).strip() for col in df.columns]
                return df
            except Exception:
                try:
                    rows = [
                        [cell.get_text(strip=True) for cell in row.find_all(['th', 'td'])]
                        for row in table_tag.find_all('tr')
                    ]
                    df = pd.DataFrame(rows[1:], columns=rows[0])
                    return df
                except Exception:
                    return pd.DataFrame()

    def table_to_list(table):
        rows = []
        if not table:
            return rows
        for row in table.find_all('tr'):
            cells = [cell.get_text(strip=True) for cell in row.find_all(['th', 'td'])]
            if cells:
                rows.append(cells)
        return rows

    def fetch_nice_rating_data(cmpCd):
        if not cmpCd:
            return {"major_grade_df": pd.DataFrame(), "special_reports": []}
        url = f"https://www.nicerating.com/disclosure/companyGradeInfo.do?cmpCd={cmpCd}"
        try:
            resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=20)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, 'html.parser')
            major_grade_table_tag = extract_table_after_marker(soup, '주요 등급내역')
            special_report_table_tag = extract_table_after_marker(soup, '스페셜 리포트')
            major_grade_df = parse_grade_table_html(major_grade_table_tag) if major_grade_table_tag else pd.DataFrame()
            special_reports = table_to_list(special_report_table_tag) if special_report_table_tag else []
            return {
                "major_grade_df": major_grade_df,
                "special_reports": special_reports,
            }
        except Exception as e:
            return {
                "major_grade_df": pd.DataFrame(),
                "special_reports": [],
                "error": f"나이스 신용평가 데이터 로드 오류: {e}"
            }

    st.markdown("---")
    st.markdown("### 📑 신용평가 보고서 및 관련 리서치")

    for cat in favorite_categories:
        for company in favorite_categories[cat]:
            kiscd = companies_map.get(company, "")
            cmpcd = config.get("cmpCD_map", {}).get(company, "")
            kr_compcd = kr_compcd_map.get(company, "")
            if not kiscd or not str(kiscd).strip():
                continue

            url_kis = f"https://www.kisrating.com/ratingsSearch/corp_overview.do?kiscd={kiscd}"
            url_nice = f"https://www.nicerating.com/disclosure/companyGradeInfo.do?cmpCd={cmpcd}"
            url_kie = f"https://www.korearatings.com/cms/frDisclosureCon/compView.do?MENU_ID=90&CONTENTS_NO=1&COMP_CD={kr_compcd}"
            with st.expander(
                f"{company} (KISCD: {kiscd} | CMP_CD: {cmpcd} | KIE_CD: {kr_compcd})", expanded=False
            ):
                st.markdown(
                    f"- [한국신용평가 (KIS)]({url_kis}) &nbsp;&nbsp; "
                    f"[나이스신용평가 (NICE)]({url_nice}) &nbsp;&nbsp; "
                    f"[한국기업평가 (KIE)]({url_kie})",
                    unsafe_allow_html=True
                )
                try:
                    resp = requests.get(url_kis, timeout=20, headers={"User-Agent": "Mozilla/5.0"})
                    if resp.status_code == 200:
                        html = resp.text
                        report_data = extract_reports_and_research(html)

                        if report_data.get("평가리포트"):
                            with st.expander("평가리포트", expanded=True):
                                st.markdown("### 한국신용평가 평가리포트")
                                df_report = pd.DataFrame(report_data["평가리포트"])
                                df_report = df_report.drop(columns=["다운로드"], errors="ignore")
                                st.dataframe(df_report)

                        if report_data.get("관련리서치"):
                            with st.expander("관련리서치", expanded=True):
                                st.markdown("### 한국신용평가 관련 리서치")
                                df_research = pd.DataFrame(report_data["관련리서치"])
                                df_research = df_research.drop(columns=["다운로드"], errors="ignore")
                                st.dataframe(df_research)

                                nice_data = fetch_nice_rating_data(cmpcd)
                                special_reports = nice_data.get("special_reports", [])
                                st.markdown("#### 나이스 신용평가 스페셜 리포트")
                                if special_reports and len(special_reports) > 1:
                                    header = special_reports[0]
                                    filtered_rows = [row for row in special_reports[1:] if len(row) == len(header)]
                                    if filtered_rows:
                                        df_special = pd.DataFrame(filtered_rows, columns=header)
                                        st.dataframe(df_special)
                                    else:
                                        st.info("표 형식이 맞는 데이터가 없습니다. (스페셜 리포트)")
                                else:
                                    st.info("스페셜 리포트 데이터가 없습니다.")
                                if nice_data.get("error"):
                                    st.warning(nice_data["error"])

                        credit_detail_list = extract_credit_details(html)
                        with st.expander("신용등급 상세정보", expanded=True):
                            if credit_detail_list:
                                st.markdown("### 한국신용평가 신용등급 상세정보")
                                df_credit_detail = pd.DataFrame(credit_detail_list)
                                st.dataframe(df_credit_detail)
                            else:
                                st.info("신용등급 상세정보가 없습니다.")

                            st.markdown("#### 나이스 신용평가 주요 등급내역")
                            nice_data = fetch_nice_rating_data(cmpcd)
                            major_grade_df = nice_data.get("major_grade_df", pd.DataFrame())
                            if not major_grade_df.empty:
                                st.dataframe(major_grade_df)
                            else:
                                st.info("주요 등급내역 데이터가 없습니다.")
                            if nice_data.get("error"):
                                st.warning(nice_data["error"])

                    else:
                        st.warning("한국신용평가 정보를 불러올 수 없습니다.")
                except Exception as e:
                    st.warning(f"신용평가 정보 파싱 오류: {e}")

                time.sleep(1)
            
def expand_keywords_with_synonyms(original_keywords):
    expanded_map = {}
    for kw in original_keywords:
        synonyms = SYNONYM_MAP.get(kw, [])
        expanded_map[kw] = [kw] + synonyms
    return expanded_map

def process_keywords_with_synonyms(favorite_to_expand_map, start_date, end_date, require_keyword_in_title=False):
    for main_kw, kw_list in favorite_to_expand_map.items():
        all_articles = []

        # 병렬 처리 시작
        with ThreadPoolExecutor(max_workers=min(5, len(kw_list))) as executor:
            futures = {
                executor.submit(fetch_naver_news, search_kw, start_date, end_date, 
                                require_keyword_in_title=require_keyword_in_title): search_kw
                for search_kw in kw_list
            }
            for future in as_completed(futures):
                search_kw = futures[future]
                try:
                    fetched = future.result()
                    # 각 기사에 검색어 정보 추가
                    fetched = [{**a, "검색어": search_kw} for a in fetched]
                    all_articles.extend(fetched)
                except Exception as e:
                    st.warning(f"{main_kw} - '{search_kw}' 검색 실패: {e}")

        # 중복 제거 여부
        if st.session_state.get("remove_duplicate_articles", False):
            all_articles = remove_duplicates(all_articles)

        st.session_state.search_results[main_kw] = all_articles
        if main_kw not in st.session_state.show_limit:
            st.session_state.show_limit[main_kw] = 5

# --- CSS 스타일 ---
st.markdown("""
<style>
[data-testid="column"] > div { gap: 0rem !important; }
.stMultiSelect [data-baseweb="tag"] { background-color: #ff5c5c !important; color: white !important; border: none !important; font-weight: bold; }
.sentiment-badge { display: inline-block; padding: 0.08em 0.6em; margin-left: 0.2em; border-radius: 0.8em; font-size: 0.85em; font-weight: bold; vertical-align: middle; }
.sentiment-positive { background: #2ecc40; color: #fff; }
.sentiment-negative { background: #ff4136; color: #fff; }
.stBox { background: #fcfcfc; border-radius: 0.7em; border: 1.5px solid #e0e2e6; margin-bottom: 1.2em; padding: 1.1em 1.2em 1.2em 1.2em; box-shadow: 0 2px 8px 0 rgba(0,0,0,0.03); }
.flex-row-bottom { display: flex; align-items: flex-end; gap: 0.5rem; margin-bottom: 0.5rem; }
.flex-grow { flex: 1 1 0%; }
.flex-btn { min-width: 90px; }
</style>

""", unsafe_allow_html=True)
st.markdown("""
<style>
.news-title { 
    word-break: break-all !important; 
    white-space: normal !important; 
    display: block !important;
    overflow: visible !important;
}
</style>
""", unsafe_allow_html=True)

def exclude_by_title_keywords(title, exclude_keywords):
    for word in exclude_keywords:
        if word in title:
            return True
    return False

def init_session_state():
    """Streamlit 세션 변수들을 일괄 초기화"""
    defaults = {
        "favorite_keywords": set(),
        "search_results": {},
        "show_limit": {},
        "search_triggered": False,
        "selected_articles": [],
        "cat_multi": [],
        "cat_major_autoset": [],
        "important_articles_preview": [],
        "important_selected_index": [],
        "article_checked_left": {},
        "article_checked": {},
        "industry_major_sub_map": {},
        "end_date": datetime.today().date(),
        "start_date": datetime.today().date() - timedelta(days=7),
        "remove_duplicate_articles": True,
        "require_exact_keyword_in_title_or_content": True,
        "filter_allowed_sources_only": False,
        "use_industry_filter": True,
        "show_sentiment_badge": False,
        "enable_summary": True,
        "keyword_input": ""
    }
    for key, default_val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_val

# --- UI 시작 ---
st.set_page_config(layout="wide")

# ✅ 세션 변수 초기화 호출
init_session_state()

col_title, col_option1, col_option2 = st.columns([0.5, 0.2, 0.3])

# --- 카테고리-산업 대분류 매핑 함수 ---
def get_industry_majors_from_favorites(selected_categories):
    favorite_to_industry_major = config["favorite_to_industry_major"]
    majors = set()
    for cat in selected_categories:
        for major in favorite_to_industry_major.get(cat, []):
            majors.add(major)
    return list(majors)

col_title, col_option1, col_option2 = st.columns([0.5, 0.2, 0.3])
with col_title:
    st.markdown(
        "<h1 style='color:#1a1a1a; margin-bottom:0.5rem;'>"
        "<a href='https://credit-issue-monitoring-news-sending.onrender.com/' target='_blank' style='text-decoration:none; color:#1a1a1a;'>"
        "📊 Credit Issue Monitoring</a></h1>",
        unsafe_allow_html=True
    )
with col_option1:
    show_sentiment_badge = st.checkbox("감성분석 배지표시", key="show_sentiment_badge")
with col_option2:
    enable_summary = st.checkbox("요약 기능", key="enable_summary")
    
col_kw_input, col_kw_btn = st.columns([0.8, 0.2])
with col_kw_input:
    keywords_input = st.text_input(label="", value="", key="keyword_input", label_visibility="collapsed")
with col_kw_btn:
    search_clicked = st.button("검색", key="search_btn", help="키워드로 검색", use_container_width=True)

st.markdown("**⭐ 산업군 선택**")
col_cat_input, col_cat_btn = st.columns([0.8, 0.2])
with col_cat_input:
    selected_categories = st.multiselect(
        "",
        list(favorite_categories.keys()), key="cat_multi", label_visibility="collapsed"
        )
if selected_categories:
    auto_selected_majors = get_industry_majors_from_favorites(selected_categories)
    st.session_state.cat_major_autoset = auto_selected_majors.copy()
else:
    st.session_state.cat_major_autoset = []
with col_cat_btn:
    category_search_clicked = st.button("🔍 검색", key="cat_search_btn", help="카테고리로 검색", use_container_width=True)
for cat in selected_categories:
    st.session_state.favorite_keywords.update(favorite_categories[cat])

# 날짜 입력 (기본 세팅: 종료일=오늘, 시작일=오늘-7일)
date_col1, date_col2 = st.columns([1, 1])
with date_col1:
    start_date = st.date_input("시작일", value=st.session_state["start_date"], key="start_date_input")
    st.session_state["start_date"] = start_date
with date_col2:
    end_date = st.date_input("종료일", value=st.session_state["end_date"], key="end_date_input")
    st.session_state["end_date"] = end_date

with st.expander("🧩 공통 필터 옵션 (항상 적용됨)"):
    for major, subs in common_filter_categories.items():
        st.markdown(f"**{major}**: {', '.join(subs)}")

with st.expander("🏭 산업별 필터 옵션 (대분류별 소분류 필터링)"):
    use_industry_filter = st.checkbox("이 필터 적용", key="use_industry_filter")

    # UI: 선택된 산업군에서 자동 매핑된 대분류 추출
    selected_major_map = get_industry_majors_from_favorites(selected_categories)

    updated_map = {}
    for major in selected_major_map:
        options = industry_filter_categories.get(major, [])
        default_selected = options if major not in st.session_state.industry_major_sub_map else st.session_state.industry_major_sub_map[major]
        selected_sub = st.multiselect(
            f"{major} 소분류 키워드",
            options,
            default=default_selected,
            key=f"subfilter_{major}_{'_'.join(selected_categories)}"
        )
        updated_map[major] = selected_sub

    st.session_state.industry_major_sub_map = updated_map
    
# --- 중복 기사 제거 기능 체크박스 포함된 키워드 필터 옵션 ---
with st.expander("🔍 키워드 필터 옵션"):
    require_exact_keyword_in_title_or_content = st.checkbox("키워드가 제목 또는 본문에 포함된 기사만 보기", key="require_exact_keyword_in_title_or_content")
    remove_duplicate_articles = st.checkbox("중복 기사 제거", key="remove_duplicate_articles", help="키워드 검색 후 중복 기사를 제거합니다.")
    filter_allowed_sources_only = st.checkbox(
        "특정 언론사만 검색", 
        key="filter_allowed_sources_only", 
        help="선택된 메이저 언론사만 필터링하고, 그 외 언론은 제외합니다."
    )

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

def detect_lang(text):
    return "ko" if re.search(r"[가-힣]", text) else "en"

def get_industry_credit_keywords():
    return """
보험사: 수익성, 자본적정성, IFRS17, K-ICS, 리스크관리, 손해율, 재보험, 유동성, 투자자산, 스트레스테스트, 경영투명성, 내부통제, 시장지위, 자금조달, 정책, 규제, 대체투자, 손익변동, 지급여력, 계약유지율, 위험집중, 체증률, 보험금지급
5대금융지주 및 은행: 자회사 신용도, 배당, 자산건전성, 정부지원, 자본비율, 유동성비율, 대손충당금, 레버리지, 스트레스, 시장위험, 금리위험, 비이자수익, 다각화, 거버넌스, 규제준수, 운영위험, 단기부채, 구조조정, 부실채권, 조기경보, 유가증권
카드사: 시장점유율, 수수료율, 대손비용, 자산건전성, 신용리스크, 대손율, 상환능력, 포트폴리오, 수익성, 거래량, 운영리스크, 법률, 파트너십, 비용, 금융조달, 신용지원, 경쟁력, 가격책정, 승인거래액, 부정사용, 결제연체
캐피탈: 사업통합, 수익안정성, 자산건전성, 해외시장, 부실률, 자금조달, 유동성, 이익창출력, 성장성, 신용리스크, 시장리스크, 법적제약, 내부통제, 채권포트폴리오, 파생상품, 그룹지원, 사업다각화, 리스크집중도, 대출채권, 부실채권비율, 회수율
지주사: 자회사 신용도, 배당안정성, 재무부담, 그룹신용, 지배구조, 재무레버리지, 부채만기, 신용지원, 수익안정성, 자본조달, 자산건전성, 현금흐름, 자본성증권, 투자리스크, 전략지원, 지분율, 내부거래, 경영권위험
에너지: 시장경쟁, 사업다각화, 해외실적, 투자규모, 가격변동성, 재무안정성, 정책변화, 환경규제, 현금흐름, 프로젝트집행, 재무파생상품리스크, 부채구조, 자본조달, 공급망, 기술전환, 글로벌경제, 탄소배출권, 에너지수급, 정부지원
발전: 전력기반, 설비투자, 전력가격, 가동률, 계약, 연료비, 부채, 자본구조, 배당정책, 재무유연성, 정부규제, 환경법규, 현금흐름, 투자계획, 차입금, 기술리스크, 사업다각화, 시장수요, 발전효율, 신재생에너지, 정부보조금
자동차: 배터리시장, 전기차수요, 설비투자, 수익성, 시장점유율, 기술경쟁력, 매출다각화, 레버리지, 고정비, 생산능력, 신제품개발, 정부정책, 공급망, 자본지출, 연구개발, 현금흐름, 성장전망, 경쟁환경, 친환경차, 관세정책
전기전자: 반도체시장, AI수요, 무역규제, 기술우위, 제품수요, 관세, 투자계획, 생산시설, 재무안정성, 연구개발, 공급망, 진입장벽, 운영효율, 환율, 보안, 가격경쟁력, 인재확보, 재무정책, 기술특허, 보안위협
소비재: 유통변화, M&A재무부담, 온라인사업, 유통채널, 브랜드, 시장점유율, 영업이익률, 현금흐름, 재무건전성, 재고관리, 경쟁압력, 혁신, 고객충성도, 비용, 공급망, 신용지원, 매출성장, 신제품런칭, 고객확보
비철철강: 수요공급, 가격변동, 해외프로젝트, 친환경설비, 비용, 자본지출, 실행력, 환경규제, 부채, 현금흐름, 시장다변화, 상품포트폴리오, 경쟁, 공급망, 기술전환, 원자재가격, 수출비중
석유화학: 경쟁력, 포트폴리오, 투자, 차입금, 세제, 재무관리, 업황민감도, 차입금비율, 자금조달, 인수합병, 수익성, 현금흐름, 자산유동화, 리스크분산, 시장점유율, 비용, 비핵심자산, 프로젝트관리, 세제혜택
특수채: 준정부기관, 보증시장, 보증사고, 자본확충, 정부지원, 신용연계, 보증잔액, 리스크, 현금성자산, 단기부채, 미회수채권, 자산건전성, 운영안정성, 보증한도, 재무안정성, 시장지위, 관리체계, 정책, 채권발행, 지급유예, 불확실성
"""
def summarize_and_sentiment_with_openai(text, do_summary=True, target_keyword=None):
    """
    본문 분석(한 줄 요약 + 시사점 + 한 줄 시사점 추가).
    target_keyword: 감성 판단의 초점을 맞출 기업/키워드
    반환: (one_line_summary, keywords, sentiment, detailed_implication, short_implication, original_text)
    """
    if not OPENAI_API_KEY:
        return "OpenAI API 키가 설정되지 않았습니다.", "", "감성 추출 실패", "", "", text
    if not text or "본문 추출 오류" in text:
        return "기사 본문이 추출 실패", "", "감성 추출 실패", "", "", text

    lang = detect_lang(text)
    industry_keywords = get_industry_credit_keywords()

    # 한국어 프롬프트만 사용, 영어 프롬프트 제거
    prompt = f"""
[산업군별 신용평가 키워드]
{industry_keywords}

아래 기사 본문을 분석해 다음 내용을 순서대로 응답하시오.
대상 기업/키워드: "{target_keyword or 'N/A'}"

1. [심층 시사점]: 단순 요약이 아니라, 신용평가사의 의견서 형식으로 이 뉴스가 해당 기업의 신용등급(상향·하향·유지), 등급 전망, 재무 건전성, 현금흐름, 유동성, 시장·규제 환경, 재무/사업 리스크에 어떤 식으로 영향을 끼칠 수 있는지 구체적으로 분석(2~3문장 이상).
2. [한 줄 시사점]: 위 시사점을 한 문장으로 요약하되, 핵심 키워드를 중심으로 해야하며, 단순 요약이 아님.
3. [한 줄 요약]: 해당 뉴스에서 기업명을 중심으로 주체, 핵심 사건, 결과를 간단하게 한 문장으로 압축.
4. [검색 키워드]: 해당 기사 검색에 사용된 키워드, 콤마로 구분.
5. [감성]: 대상 기업에 대한 긍정 또는 부정 중 하나만.
6. [주요 키워드]: 인물, 기업, 조직명만 콤마(,)로, 없으면 없음

특히 [심층 시사점]에서는 아래 사항을 필수로 포함:
- 등급 변동을 유발할 수 있는 직접적/간접적 사건 및 재무 지표 변화
- 기업의 정책/시장/사업환경 변화에 따른 신용 리스크 요인과 등급 방향성
- 동종업계나 과거 사례와 비교되는 차별화 지점(있으면 명시)
- 단순 현상보고(한줄 요약)와 명확히 구분되는 신용평가사의 '심층 의견'을 2~3문장 이상으로 서술


[기사 본문]
{text}
"""
    role_content = "너는 신용평가 전문가이자 금융 뉴스 분석가이다. 정확하고 명확하게 분석하라."

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": role_content},
                {"role": "user", "content": prompt}
            ],
            max_tokens=900,
            temperature=0
        )
        answer = response.choices[0].message.content.strip()
    except Exception as e:
        return f"요약 오류: {e}", "", "감성 추출 실패", "", "", text

    # 정규식으로 결과 추출
    def extract_group(tag):
        pattern = rf"\[{tag}\]:\s*([\s\S]+?)(?=\n\[\w+\]:|\n\d\. \[|$)"
        m = re.search(pattern, answer)
        return m.group(1).strip() if m else ""

    detailed_implication = extract_group("심층 시사점") or "시사점 추출 실패"
    short_implication = extract_group("한 줄 시사점") or "한 줄 시사점 요약 실패"
    one_line = extract_group("한 줄 요약") or "요약 추출 실패"
    keywords = extract_group("검색 키워드") or ""
    sentiment = extract_group("감성") or "감성 추출 실패"
    if sentiment.lower() == "positive" or sentiment == "긍정":
        sentiment = "긍정"
    elif sentiment.lower() == "negative" or sentiment == "부정":
        sentiment = "부정"
    else:
        sentiment = "감성 추출 실패"
    key_entities = extract_group("주요 키워드") or ""

    return one_line, keywords, sentiment, detailed_implication, short_implication, text

def infer_source_from_url(url):
    domain = urlparse(url).netloc
    if domain.startswith("www."):
        domain = domain[4:]
    return domain

NAVER_CLIENT_ID = os.environ.get("NAVER_CLIENT_ID")
NAVER_CLIENT_SECRET = os.environ.get("NAVER_CLIENT_SECRET")
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")

class Telegram:
    def __init__(self):
        self.bot = telepot.Bot(TELEGRAM_TOKEN)  # 이미 환경변수 기반
        self.chat_id = TELEGRAM_CHAT_ID
    def send_message(self, message):
        self.bot.sendMessage(self.chat_id, message, parse_mode="Markdown", disable_web_page_preview=True)
        
def filter_by_issues(title, desc, selected_keywords, require_keyword_in_title=False):
    if require_keyword_in_title and selected_keywords:
        if not any(kw.lower() in title.lower() for kw in selected_keywords):
            return False
    return True

def fetch_naver_news(query, start_date=None, end_date=None, limit=1000, require_keyword_in_title=False):
    headers = {
        "X-Naver-Client-Id": NAVER_CLIENT_ID,
        "X-Naver-Client-Secret": NAVER_CLIENT_SECRET
    }
    articles = []
    for start in range(1, 1001, 100):
        if len(articles) >= limit:
            break
        params = {
            "query": query,
            "display": 100,
            "start": start,
            "sort": "date"
        }
        response = requests.get("https://openapi.naver.com/v1/search/news.json", headers=headers, params=params)
        if response.status_code != 200:
            break

        items = response.json().get("items", [])
        for item in items:
            title = html.unescape(re.sub("<.*?>", "", item["title"]))
            desc  = html.unescape(re.sub("<.*?>", "", item["description"]))
            pub_date = datetime.strptime(item["pubDate"], "%a, %d %b %Y %H:%M:%S %z").date()

            if start_date and pub_date < start_date:
                continue
            if end_date and pub_date > end_date:
                continue
            if not filter_by_issues(title, desc, [query], require_keyword_in_title):
                continue
            if exclude_by_title_keywords(title, EXCLUDE_TITLE_KEYWORDS):
                continue

            source = item.get("source") or infer_source_from_url(item.get("originallink", "")) or "Naver"
            source_domain = source.lower()
            if source_domain.startswith("www."):
                source_domain = source_domain[4:]

            real_link = item.get("originallink") or item["link"]

            articles.append({
                "title": title,
                "description": desc,  # 혹시 엑셀에 설명도 쓸 경우 대비
                "link": real_link,
                "date": pub_date.strftime("%Y-%m-%d"),
                "source": source_domain
            })

        if len(items) < 100:
            break
    return articles[:limit]

def process_keywords(keyword_list, start_date, end_date, require_keyword_in_title=False):
    for k in keyword_list:
        articles = fetch_naver_news(k, start_date, end_date, require_keyword_in_title=require_keyword_in_title)
        st.session_state.search_results[k] = articles
        if k not in st.session_state.show_limit:
            st.session_state.show_limit[k] = 5

def summarize_article_from_url(article_url, title, do_summary=True, target_keyword=None, description=None):
    cache_key_base = re.sub(r"\W+", "", article_url)[-16:]
    summary_key = f"summary_{cache_key_base}"

    if summary_key in st.session_state:
        return st.session_state[summary_key]

    try:
        full_text = extract_article_text(article_url, fallback_desc=description, fallback_title=title)
        if full_text.startswith("본문 추출 오류"):
            result = (full_text, "", "감성 추출 실패", "", "", full_text)  # 6개 요소 맞춤
        else:
            one_line, summary, sentiment, implication, short_implication, text = summarize_and_sentiment_with_openai(
                full_text, do_summary=do_summary, target_keyword=target_keyword
            )
            result = (one_line, summary, sentiment, implication, short_implication, text)
    except Exception as e:
        result = (f"요약 오류: {e}", "", "감성 추출 실패", "", "", "")

    st.session_state[summary_key] = result
    return result

def or_keyword_filter(article, *keyword_lists):
    text = (article.get("title", "") + " " + article.get("description", "") + " " + article.get("full_text", ""))
    for keywords in keyword_lists:
        if any(kw in text for kw in keywords if kw):
            return True
    return False

def article_contains_exact_keyword(article, keywords):
    title = article.get("title", "")
    content = ""
    cache_key = article.get("link", "")
    summary_cache_key = None
    for key in st.session_state.keys():
        if key.startswith("summary_") and cache_key in key:
            summary_cache_key = key
            break
    if summary_cache_key and isinstance(st.session_state[summary_cache_key], tuple):
        _, _, _, content = st.session_state[summary_cache_key]
    for kw in keywords:
        if kw and (kw in title or (content and kw in content)):
            return True
    return False

def article_passes_all_filters(article):
    # 제목에 제외 키워드가 포함되면 제외
    if exclude_by_title_keywords(article.get('title', ''), EXCLUDE_TITLE_KEYWORDS):
        return False

    # 날짜 범위 필터링
    try:
        pub_date = datetime.strptime(article['date'], '%Y-%m-%d').date()
        if pub_date < st.session_state.get("start_date") or pub_date > st.session_state.get("end_date"):
            return False
    except:
        return False

    # 키워드 필터: 입력 키워드 및 카테고리 키워드 집합 준비
    all_keywords = []
    if "keyword_input" in st.session_state:
        all_keywords.extend([k.strip() for k in st.session_state["keyword_input"].split(",") if k.strip()])
    if "cat_multi" in st.session_state:
        for cat in st.session_state["cat_multi"]:
            all_keywords.extend(favorite_categories.get(cat, []))

    # 키워드 필터(입력 및 카테고리 키워드) 통과 여부
    keyword_passed = article_contains_exact_keyword(article, all_keywords)

    # 언론사 도메인 필터링 (특정 언론사만 필터링)
    if st.session_state.get("filter_allowed_sources_only", True):
        source = article.get('source', '').lower()
        if source.startswith("www."):
            source = source[4:]
        if source not in ALLOWED_SOURCES:
            return False

    # 공통 필터 조건 (AND 조건, 즉 반드시 통과해야 함)
    common_passed = or_keyword_filter(article, ALL_COMMON_FILTER_KEYWORDS)
    if not common_passed:
        return False

    # 산업별 필터 조건 (OR 조건)
    industry_passed = True
    if st.session_state.get("use_industry_filter", False):
        keyword = article.get("키워드")  # 회사명 또는 키워드 항목명
        matched_major = None
        for cat, companies in favorite_categories.items():
            if keyword in companies:
                majors = get_industry_majors_from_favorites([cat])
                if majors:
                    matched_major = majors[0]
                    break
        if matched_major:
            sub_keyword_filter = st.session_state.industry_major_sub_map.get(matched_major, [])
            if sub_keyword_filter:
                industry_passed = or_keyword_filter(article, sub_keyword_filter)

    # 최종 필터링: 공통 필터는 반드시 통과하고,
    # 산업별 필터나 키워드 필터 중 하나라도 통과하면 통과
    if not (industry_passed or keyword_passed):
        return False

    return True

# --- 중복 기사 제거 함수 ---
def is_similar(title1, title2, threshold=0.5):
    ratio = difflib.SequenceMatcher(None, title1, title2).ratio()
    return ratio >= threshold

def remove_duplicates(articles):
    unique_articles = []
    titles = []
    for article in articles:
        title = article.get("title", "")
        if all(not is_similar(title, existing_title) for existing_title in titles):
            unique_articles.append(article)
            titles.append(title)
    return unique_articles

# 항상 먼저 선언해 에러 방지
keyword_list = [k.strip() for k in keywords_input.split(",") if k.strip()] if keywords_input else []
search_clicked = False

if keyword_list:
        search_clicked = True

if keyword_list and (search_clicked or st.session_state.get("search_triggered")):
    with st.spinner("뉴스 검색 중..."):
        # 동의어 확장
        expanded = expand_keywords_with_synonyms(sorted(keyword_list))
        process_keywords_with_synonyms(
            expanded,
            st.session_state["start_date"],
            st.session_state["end_date"],
            require_keyword_in_title=st.session_state.get("require_exact_keyword_in_title_or_content", False)
        )
    st.session_state.search_triggered = False


if category_search_clicked and selected_categories:
    with st.spinner("뉴스 검색 중..."):
        keywords = set()
        for cat in selected_categories:
            keywords.update(favorite_categories[cat])

        expanded = expand_keywords_with_synonyms(sorted(keywords))
        process_keywords_with_synonyms(
            expanded,
            st.session_state["start_date"],
            st.session_state["end_date"],
            require_keyword_in_title=st.session_state.get("require_exact_keyword_in_title_or_content", False)
        )

def safe_title(val):
    if pd.isnull(val) or str(val).strip() == "" or str(val).lower() == "nan" or str(val) == "0":
        return "제목없음"
    return str(val)

def clean_excel_formula_text(text):
    """엑셀 수식(HYPERLINK)에서 깨짐 방지용 전처리"""
    if not isinstance(text, str):  # None이나 숫자이면 문자 변환
        text = str(text)
    text = text.replace('"', "'")   # 큰따옴표 → 홑따옴표
    text = text.replace('\n', ' ')  # 줄바꿈 → 공백
    text = text.replace('\r', '')
    return text[:250]  # 안전하게 255자 미만으로 제한

def get_excel_download_with_favorite_and_excel_company_col(summary_data, favorite_categories, excel_company_categories, search_results):
    import pandas as pd
    from io import BytesIO

    def clean_text(text):
        if not isinstance(text, str):
            text = str(text)
        text = text.replace('"', "'").replace('\n', ' ').replace('\r', '')
        return text[:200]

    # 회사 리스트 (중복 제거하며 순서 유지)
    sector_list = []
    for cat in favorite_categories:
        sector_list.extend(favorite_categories[cat])
    sector_list = list(dict.fromkeys(sector_list))

    # 각 회사에 대응하는 엑셀 표기명 리스트
    excel_sector_list = []
    for cat in excel_company_categories:
        excel_sector_list.extend(excel_company_categories[cat])
    excel_sector_list = list(dict.fromkeys(excel_sector_list))

    # 빈 DataFrame일 경우 대비
    if summary_data is None or len(summary_data) == 0:
        df_empty = pd.DataFrame(columns=["기업명", "표기명", "건수", "중요뉴스1", "중요뉴스2", "시사점"])
        output = BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df_empty.to_excel(writer, index=False, sheet_name='뉴스요약')
            worksheet = writer.sheets['뉴스요약']
            worksheet.set_column(0, 5, 30)
        output.seek(0)
        return output

    df = pd.DataFrame(summary_data)

    # ‘한줄시사점’ 우선, 없으면 ‘시사점’, ‘implication’ 컬럼으로 설정
    if "한줄시사점" in df.columns:
        implication_col = "한줄시사점"
    elif "시사점" in df.columns:
        implication_col = "시사점"
    elif "implication" in df.columns:
        implication_col = "implication"
    else:
        implication_col = None

    # 키워드 관련 컬럼명 결정
    if "키워드" in df.columns:
        keyword_col = "키워드"
    elif "기업명" in df.columns:
        keyword_col = "기업명"
    elif "회사명" in df.columns:
        keyword_col = "회사명"
    else:
        keyword_col = df.columns[0] if len(df.columns) > 0 else "기업명"

    rows = []
    for idx, company in enumerate(sector_list):
        # 해당 회사 관련 모든 기사 리스트 추출
        search_articles = search_results.get(company, [])

        # 공통 필터와 산업별 필터 통과 기사만 필터링 (필요시 산업별 필터 조건 추가)
        filtered_articles = []
        for article in search_articles:
            passes_common = any(kw in (article.get("title", "") + article.get("description", "")) for kw in ALL_COMMON_FILTER_KEYWORDS)
            passes_industry = True
            # 필요 시 산업별 필터링 로직 추가 가능

            if passes_common and passes_industry:
                filtered_articles.append(article)

        # 중복 기사 제거 옵션 적용
        if st.session_state.get("remove_duplicate_articles", False):
            filtered_articles = remove_duplicates(filtered_articles)

        total_count = len(filtered_articles)

        # 해당 회사의 요약 데이터(중복 제거, 필터링된) 중 최신 2개 기사 추출
        filtered_df = df[df.get(keyword_col, "") == company].sort_values(by='날짜', ascending=False)
        hl_news = ["", ""]
        implications = ["", ""]
        for i, art in enumerate(filtered_df.itertuples()):
            if i > 1:
                break
            date_val = getattr(art, "날짜", "") or ""
            title_val = getattr(art, "기사제목", "") or getattr(art, "제목", "")
            link_val = getattr(art, "링크", "") or getattr(art, "link", "")
            display_text = f"({clean_text(date_val)}){clean_text(title_val)}"
            if title_val and link_val:
                hl_news[i] = f'=HYPERLINK("{clean_text(link_val)}", "{display_text}")'
            else:
                hl_news[i] = display_text or ""

            if implication_col:
                implications[i] = getattr(art, implication_col, "") or ""
            else:
                implications[i] = ""

        # ‘한줄 시사점’을 번호 매겨 줄바꿈으로 병합 (최대 2개)
        merged_implication = ""
        if implications[0]:
            merged_implication += f"1. {implications[0]}"
        if implications[1]:
            if merged_implication:
                merged_implication += f"\n2. {implications[1]}"
            else:
                merged_implication = f"2. {implications[1]}"

        rows.append({
            "기업명": company,
            "표기명": excel_sector_list[idx] if idx < len(excel_sector_list) else "",
            "건수": total_count,
            "중요뉴스1": hl_news[0],
            "중요뉴스2": hl_news[1],
            "시사점": merged_implication
        })

    result_df = pd.DataFrame(rows, columns=["기업명", "표기명", "건수", "중요뉴스1", "중요뉴스2", "시사점"])

    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        result_df.to_excel(writer, index=False, sheet_name='뉴스요약')
        worksheet = writer.sheets['뉴스요약']
        for i, col in enumerate(result_df.columns):
            worksheet.set_column(i, i, 30)
    output.seek(0)
    return output

def generate_important_article_list(search_results, common_keywords, industry_keywords, favorites):
    import os
    from openai import OpenAI
    import re

    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
    client = OpenAI(api_key=OPENAI_API_KEY)
    result = []

    # 기존 함수 내에 섹터별 키워드 파싱 처리 포함
    def parse_industry_keywords():
        raw_text = get_industry_credit_keywords()
        industry_dict = {}
        for line in raw_text.strip().split("\n"):
            if ":" in line:
                sector, keywords = line.split(":", 1)
                industry_dict[sector.strip()] = [kw.strip() for kw in keywords.split(",") if kw.strip()]
        return industry_dict

    industry_keywords_dict = parse_industry_keywords()

    for category, companies in favorites.items():
        # 카테고리명(category)에 해당하는 섹터 키워드 얻기
        sector_keywords = industry_keywords_dict.get(category, [])

        for comp in companies:
            articles = search_results.get(comp, [])

            # 섹터 핵심 키워드가 기사 내 포함된 경우만 필터링
            target_articles = []
            for a in articles:
                text = (a.get("title", "") + " " + a.get("description", "")).lower()
                if any(kw.lower() in text for kw in sector_keywords):
                    target_articles.append(a)

            if not target_articles:
                continue

            prompt_list = "\n".join([f"{i+1}. {a['title']} - {a['link']}" for i, a in enumerate(target_articles)])

            prompt = (
                f"[기사 목록]\n{prompt_list}\n\n"
                f"분석의 초점은 반드시 '{comp}' 기업(또는 키워드)이며, "
                f"'{category}' 산업의 신용평가 핵심 이슈 키워드({', '.join(sector_keywords[:10])}...)가 포함된 뉴스 중\n"
                "신용 평가 관점에서 중요한 뉴스 2건을 선정해 주세요.\n"
                "감성 판단은 필요 없으며, 중요도에 따라 자유롭게 선정하면 됩니다.\n\n"
                "선정한 뉴스를 각각 별도의 행으로 작성하세요.\n\n"
                "[중요 뉴스 1]: (중요 뉴스 제목)\n"
                "[중요 뉴스 2]: (중요 뉴스 제목)\n"
            )

            try:
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=900,
                    temperature=0
                )
                answer = response.choices[0].message.content.strip()
                news1_match = re.search(r"\[중요 뉴스 1\]:\s*(.+)", answer)
                news2_match = re.search(r"\[중요 뉴스 2\]:\s*(.+)", answer)
                
                news1_title = news1_match.group(1).strip() if news1_match else ""
                news2_title = news2_match.group(1).strip() if news2_match else ""
                
                for a in target_articles:
                    if news1_title and news1_title in a["title"]:
                        result.append({
                            "키워드": comp,
                            "기사제목": a["title"],
                            "링크": a["link"],
                            "날짜": a["date"],
                            "출처": a["source"],
                            "시사점": "",  # 필요시 빈 문자열로 처리
                        })
                    if news2_title and news2_title in a["title"]:
                        result.append({
                            "키워드": comp,
                            "기사제목": a["title"],
                            "링크": a["link"],
                            "날짜": a["date"],
                            "출처": a["source"],
                            "시사점": "",  # 필요시 빈 문자열로 처리
                        })
            except Exception:
                continue

    return result

def extract_article_text(url, fallback_desc=None, fallback_title=None):
    """
    뉴스 기사 본문을 최대한 정확하게 추출
    url: 기사 원문 URL
    fallback_desc, fallback_title: 본문 추출 실패시 사용할 검색 API의 요약/제목
    """
    # 포털 뉴스 차단
    PORTAL_DOMAINS = ["news.naver.com", "n.news.naver.com", "news.daum.net"]
    if any(domain in url for domain in PORTAL_DOMAINS):
        return f"본문 추출 오류: 포털 뉴스 중계 URL입니다. originallink 사용 권장."

    try:
        # 1차 시도: newspaper3k
        article = newspaper.Article(url, language='ko')
        article.download()
        article.parse()
        text = article.text.strip()

        # 불필요 문구 제거
        text = re.sub(r"\S+@\S+", "", text)  # 이메일 제거
        text = re.sub(r"▶.*", "", text)      # '▶'로 시작하는 행 제거
        text = re.sub(r"(무단전재\s*및\s*재배포\s*금지.*$)", "", text)

        # 2차: 텍스트 길이 검증 (글자가 너무 짧으면 fallback)
        if len(text) < 100 and fallback_desc:
            # 너무 짧으면 설명(description) 붙여서 보완
            text = text + "\n\n" + fallback_desc
        
        return text

    except Exception as e:
        # 2차 시도: 직접 HTML 파싱
        try:
            resp = requests.get(url, timeout=5, headers={"User-Agent": "Mozilla/5.0"})
            soup = BeautifulSoup(resp.text, "html.parser")
            # 대표적인 한국 언론 본문 영역 선택자
            selectors = [
                "div#articleBodyContents", 
                "div.article_body", 
                "div#newsEndContents",
                "div[itemprop='articleBody']"
            ]
            for sel in selectors:
                body = soup.select_one(sel)
                if body:
                    # 텍스트 정제
                    text = " ".join(body.get_text(separator=" ").split())
                    text = re.sub(r"\S+@\S+", "", text)
                    if len(text) > 200:
                        return text
        except Exception:
            pass

        # fallback: 제목 + 설명이라도 제공
        if fallback_title or fallback_desc:
            return f"[기사제목] {fallback_title or ''}\n[요약정보] {fallback_desc or ''}"

        return f"본문 추출 오류: {e}"
    
def extract_keyword_from_link(search_results, article_link):
    """
    뉴스검색결과 dict와 기사 링크로 해당 기사의 키워드(회사명/카테고리)를 추출
    """
    for kw, arts in search_results.items():
        for art in arts:
            if art.get("link") == article_link:
                return kw
    return ""

def build_important_excel_format(important_articles, favorite_categories, excel_categories, search_results):
    import pandas as pd

    df = pd.DataFrame(important_articles)

    # 회사 리스트 (중복 제거하며 순서 유지)
    sector_list = []
    for cat in favorite_categories:
        sector_list.extend(favorite_categories[cat])
    sector_list = list(dict.fromkeys(sector_list))

    excel_sector_list = []
    for cat in excel_categories:
        excel_sector_list.extend(excel_categories[cat])
    excel_sector_list = list(dict.fromkeys(excel_sector_list))

    rows = []

    for idx, company in enumerate(sector_list):
        # 기사 필터링 및 중복 제거
        all_articles = search_results.get(company, [])

        filtered_articles = []
        for art in all_articles:
            if article_passes_filters(art):  # 또는 article_passes_filters(art) 함수에 맞게 변경
                filtered_articles.append(art)

        if 'remove_duplicate_articles' in st.session_state and st.session_state['remove_duplicate_articles']:
            filtered_articles = remove_duplicates(filtered_articles)

        total_count = len(filtered_articles)

        # 해당 회사의 선택된 중요기사 요약 데이터(이미 중복 제거, 필터링된)를 가져옴
        filtered_df = df[df['기업명'] == company].sort_values(by='날짜', ascending=False)

        hl_news = []
        for i, art in enumerate(filtered_df.itertuples()):
            if i > 1:
                break
            title = getattr(art, '제목', '') or ''
            link = getattr(art, '링크', '') or ''
            if title and link:
                hl_news.append(f'=HYPERLINK("{link}", "{title}")')
            else:
                hl_news.append(title or '')
        # 2개까지 채우고 부족하면 빈문자열 채움
        while len(hl_news) < 2:
            hl_news.append('')

        # 시사점 병합 (최대 2개)
        implication_col = '시사점' if '시사점' in df.columns else ('implication' if 'implication' in df.columns else None)
        implications = []
        for i, art in enumerate(filtered_df.itertuples()):
            if i > 1:
                break
            val = getattr(art, implication_col, '') if implication_col else ''
            implications.append(val)
        merged_implication = ''
        if implications:
            merged_implication = '\n'.join(f"{idx+1}. {txt}" for idx, txt in enumerate(implications) if txt)

        rows.append({
            '기업명': company,
            '표기명': excel_sector_list[idx] if idx < len(excel_sector_list) else '',
            '건수': total_count,
            '중요뉴스1': hl_news[0],
            '중요뉴스2': hl_news[1],
            '시사점': merged_implication
        })

    result_df = pd.DataFrame(rows, columns=['기업명', '표기명', '건수', '중요뉴스1', '중요뉴스2', '시사점'])

    from io import BytesIO
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        result_df.to_excel(writer, index=False, sheet_name='뉴스요약')
        worksheet = writer.sheets['뉴스요약']
        for i, col in enumerate(result_df.columns):
            worksheet.set_column(i, i, 30)
    output.seek(0)
    return output
   
def matched_filter_keywords(article, common_keywords, industry_keywords):
    """
    기사 제목/요약/본문에서 실제로 포함된 필터 키워드 리스트 반환
    """
    text_candidates = [
        article.get("title", ""),
        article.get("description", ""),
        article.get("요약본", ""),
        article.get("요약", ""),
        article.get("full_text", ""),
        article.get("content", ""),
    ]
    text_long = " ".join([str(t) for t in text_candidates if t])
    matched_common = [kw for kw in common_keywords if kw in text_long]
    matched_industry = [kw for kw in industry_keywords if kw in text_long]
    return list(set(matched_common + matched_industry))

def render_articles_with_single_summary_and_telegram(
    results, show_limit, show_sentiment_badge=True, enable_summary=True
):
    SENTIMENT_CLASS = {"긍정": "sentiment-positive", "부정": "sentiment-negative"}
    col_list, col_summary = st.columns([1, 1])

    # ---------------------------- 뉴스 목록 열 ---------------------------- #
    with col_list:
        st.markdown("### 🔍 뉴스 검색 결과")
        for category_name, company_list in favorite_categories.items():
            companies_with_results = [c for c in company_list if c in results]
            if not companies_with_results:
                continue
            with st.expander(f"📂 {category_name}", expanded=True):
                for company in companies_with_results:
                    articles = results[company]
                    with st.expander(f"[{company}] ({len(articles)}건)", expanded=False):
                        all_article_keys = []
                        for idx, article in enumerate(articles):
                            uid = re.sub(r"\W+", "", article["link"])[-16:]
                            key = f"{company}_{idx}_{uid}"
                            all_article_keys.append(key)
                        prev_value = all(st.session_state.article_checked.get(k, False) for k in all_article_keys)
                        select_all = st.checkbox(
                            f"전체 기사 선택/해제 ({company})",
                            value=prev_value,
                            key=make_key("selectall", category_name, company)
                        )
                        if select_all != prev_value:
                            for k in all_article_keys:
                                st.session_state.article_checked[k] = select_all
                                st.session_state.article_checked_left[k] = select_all
                            st.rerun()

                        for idx, article in enumerate(articles):
                            uid = re.sub(r"\W+", "", article["link"])[-16:]
                            key = f"{company}_{idx}_{uid}"
                            cache_key = f"summary_{key}"
                            cols = st.columns([0.04, 0.96])
                            with cols[0]:
                                checked = st.checkbox(
                                    "",
                                    value=st.session_state.article_checked.get(key, False),
                                    key=f"news_{key}",
                                )
                            with cols[1]:
                                sentiment = ""
                                if show_sentiment_badge and cache_key in st.session_state:
                                    _, _, sentiment, _, _ = st.session_state[cache_key]
                                badge_html = (
                                    f"<span class='sentiment-badge {SENTIMENT_CLASS.get(sentiment, 'sentiment-negative')}'>{sentiment}</span>"
                                    if sentiment else ""
                                )
                                search_word_info = f" | 검색어: {article.get('검색어', '')}" if article.get("검색어") else ""
                                st.markdown(
                                    f"<span class='news-title'><a href='{article['link']}' target='_blank'>{article['title']}</a></span> "
                                    f"{badge_html} {article['date']} | {article['source']}{search_word_info}",
                                    unsafe_allow_html=True,
                                )
                            st.session_state.article_checked_left[key] = checked
                            st.session_state.article_checked[key] = checked

    # ---------------------------- 선택 기사 요약/감성분석 열 ---------------------------- #
    with col_summary:
        st.markdown("### 선택된 기사 요약/감성분석")
        with st.container(border=True):
            industry_keywords_all = []
            if st.session_state.get("use_industry_filter", False):
                for sublist in st.session_state.industry_major_sub_map.values():
                    industry_keywords_all.extend(sublist)

            grouped_selected = {}
            for cat_name, company_list in favorite_categories.items():
                for company in company_list:
                    if company in results:
                        for idx, article in enumerate(results[company]):
                            uid = re.sub(r"\W+", "", article["link"])[-16:]
                            key = f"{company}_{idx}_{uid}"
                            if st.session_state.article_checked.get(key, False):
                                grouped_selected.setdefault(cat_name, {}).setdefault(company, []).append(
                                    (company, idx, article)
                                )

            def process_article(item):
                keyword, idx, art = item
                cache_key = f"summary_{keyword}_{idx}_" + re.sub(r"\W+", "", art["link"])[-16:]
                if cache_key in st.session_state:
                    one_line, summary, sentiment, implication, short_implication, full_text = st.session_state[cache_key]
                else:
                    one_line, summary, sentiment, implication, short_implication, full_text = summarize_article_from_url(
                        art["link"], art["title"], do_summary=enable_summary, target_keyword=keyword
                    )
                    st.session_state[cache_key] = (one_line, summary, sentiment, implication, short_implication, full_text)
                filter_hits = matched_filter_keywords(
                    {"title": art["title"], "요약본": summary, "요약": one_line, "full_text": full_text},
                    ALL_COMMON_FILTER_KEYWORDS,
                    industry_keywords_all
                )
                return {
                    "키워드": keyword,
                    "필터히트": ", ".join(filter_hits),
                    "기사제목": safe_title(art["title"]),
                    "요약": one_line,
                    "요약본": summary,
                    "감성": sentiment,
                    "시사점": implication,
                    "한줄시사점": short_implication,  
                    "링크": art["link"],
                    "날짜": art["date"],
                    "출처": art["source"],
                    "full_text": full_text or "",
                }

            from concurrent.futures import ThreadPoolExecutor
            for cat_name, comp_map in grouped_selected.items():
                for company, items in comp_map.items():
                    with ThreadPoolExecutor(max_workers=10) as executor:
                        grouped_selected[cat_name][company] = list(executor.map(process_article, items))

            total_selected_count = 0
            for cat_name, comp_map in grouped_selected.items():
                with st.expander(f"📂 {cat_name}", expanded=True):
                    for company, arts in comp_map.items():
                        with st.expander(f"[{company}] ({len(arts)}건)", expanded=True):
                            for art in arts:
                                total_selected_count += 1
                                st.markdown(
                                    f"#### <span class='news-title'><a href='{art['링크']}' target='_blank'>{art['기사제목']}</a></span> "
                                    f"<span class='sentiment-badge {SENTIMENT_CLASS.get(art['감성'], 'sentiment-negative')}'>{art['감성']}</span>",
                                    unsafe_allow_html=True
                                )
                                st.markdown(f"- **검색 키워드:** `{art['키워드']}`")
                                st.markdown(f"- **필터로 인식된 키워드:** `{art['필터히트'] or '없음'}`")
                                st.markdown(f"- **날짜/출처:** {art['날짜']} | {art['출처']}")
                                if enable_summary:
                                    st.markdown(f"- **한 줄 요약:** {art['요약']}")
                                    st.markdown(f"- **한 줄 시사점:** {art.get('한줄시사점', '없음')}")
                                    st.markdown(f"- **시사점:** {art['시사점'] or '없음'}")
                                st.markdown(f"- **감성분석:** `{art['감성']}`")
                                st.markdown("---")

            st.session_state.selected_articles = [
                art for comp_map in grouped_selected.values() for arts in comp_map.values() for art in arts
            ]
            st.write(f"선택된 기사 개수: {total_selected_count}")

            # 다운로드 / 전체 해제
            col_dl1, col_dl2 = st.columns([0.55, 0.45])
            with col_dl1:
                st.download_button(
                    label="📥 맞춤 엑셀 다운로드",
                    data=get_excel_download_with_favorite_and_excel_company_col(
                        st.session_state.selected_articles,
                        favorite_categories,
                        excel_company_categories,
                        st.session_state.search_results
                    ).getvalue(),
                    file_name="뉴스요약_맞춤형.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
            with col_dl2:
                if st.button("🗑 선택 해제 (전체)"):
                    for key in list(st.session_state.article_checked.keys()):
                        st.session_state.article_checked[key] = False
                    for key in list(st.session_state.article_checked_left.keys()):
                        st.session_state.article_checked_left[key] = False
                    st.rerun()

        render_important_article_review_and_download()

def render_important_article_review_and_download():
    import re
    from collections import defaultdict
    import streamlit as st

    with st.container(border=True):
        st.markdown("### ⭐ 중요 기사 리뷰 및 편집")

        auto_btn = st.button("🚀 OpenAI 기반 중요 기사 자동 선정")
        if auto_btn:
            with st.spinner("OpenAI로 중요 뉴스 선정 중..."):
                filtered_results_for_important = {}
                for keyword, articles in st.session_state.search_results.items():
                    filtered_articles = [a for a in articles if article_passes_all_filters(a)]
                    if st.session_state.get("remove_duplicate_articles", False):
                        filtered_articles = remove_duplicates(filtered_articles)
                    if filtered_articles:
                        filtered_results_for_important[keyword] = filtered_articles

                important_articles = generate_important_article_list(
                    search_results=filtered_results_for_important,
                    common_keywords=ALL_COMMON_FILTER_KEYWORDS,
                    industry_keywords=st.session_state.get("industry_sub", []),
                    favorites=favorite_categories
                )
                # key 명 통일 및 시사점 필드 포함 (시사점은 빈 문자열로 초기화, 필요 시 OpenAI 결과 반영 가능)
                for i, art in enumerate(important_articles):
                    important_articles[i] = {
                        "키워드": art.get("키워드") or art.get("회사명") or art.get("keyword") or "",
                        "기사제목": art.get("기사제목") or art.get("제목") or art.get("title") or "",
                        "감성": art.get("감성", ""),
                        "링크": art.get("링크") or art.get("link", ""),
                        "날짜": art.get("날짜") or art.get("date", ""),
                        "출처": art.get("출처") or art.get("source", ""),
                        "시사점": art.get("시사점", "")  # 시사점 필드 추가 (자동선정 시 채워질 수 있음)
                    }
                st.session_state["important_articles_preview"] = important_articles
                st.session_state["important_selected_index"] = []

        articles = st.session_state.get("important_articles_preview", [])
        selected_indexes = st.session_state.get("important_selected_index", [])

        # 대분류(major) - 소분류(minor) 그룹화
        major_map = defaultdict(lambda: defaultdict(list))  # major_map[대분류][소분류] = [기사 리스트]
        for art in articles:
            keyword = art.get("키워드") or art.get("회사명") or ""
            found_major = None
            for major, minors in favorite_categories.items():
                if keyword in minors:
                    found_major = major
                    break
            if found_major:
                major_map[found_major][keyword].append(art)

        st.markdown("🎯 **중요 기사 목록 (교체 또는 삭제할 항목을 체크하세요)**")

        from concurrent.futures import ThreadPoolExecutor
        one_line_map = {}
        to_summarize = []

        for major, minor_map in major_map.items():
            for minor, arts in minor_map.items():
                for idx, article in enumerate(arts):
                    link = article.get("링크", "")
                    cleaned_id = re.sub(r"\W+", "", link)[-16:] if link else ""
                    cache_hit = False
                    for k, v in st.session_state.items():
                        if k.startswith("summary_") and cleaned_id in k and isinstance(v, tuple):
                            one_line_map[(major, minor, idx)] = v
                            cache_hit = True
                            break
                    if not cache_hit and link:
                        to_summarize.append((major, minor, idx, link, article.get("기사제목", "")))

        if to_summarize:
            with st.spinner("중요 기사 요약 생성 중..."):
                def get_one_line(args):
                    major, minor, idx, link, title = args
                    one_line, summary, sentiment, implication, short_implication, full_text = summarize_article_from_url(link, title, do_summary=True)
                    return (major, minor, idx), (one_line, summary, sentiment, implication, short_implication, full_text)

                with ThreadPoolExecutor(max_workers=10) as executor:
                    for key, data_tuple in executor.map(get_one_line, to_summarize):
                        one_line_map[key] = data_tuple

        new_selection = []
        for major, minor_map in major_map.items():
            with st.expander(f"📊 {major}", expanded=True):
                for minor, arts in minor_map.items():
                    with st.expander(f"{minor} ({len(arts)}건)", expanded=False):
                        for idx, article in enumerate(arts):
                            check_key = f"important_chk_{major}_{minor}_{idx}"
                            # 한 줄에 체크박스 + 감성 + 기사제목 하이퍼링크 배치
                            cols = st.columns([0.06, 0.94])
                            with cols[0]:
                                checked = st.checkbox(
                                "",
                                key=check_key,
                                value=(check_key in selected_indexes)
                            )
                            if checked:
                                new_selection.append((major, minor, idx))
                        with cols[1]:
                            st.markdown(
                                f"{article.get('감성','')} | <a href='{article.get('링크','')}' target='_blank'>{article.get('기사제목','제목없음')}</a>",
                                unsafe_allow_html=True
                            )

                            # 시사점 및 한줄 시사점 출력
                            summary_data = one_line_map.get((major, minor, idx))
                            implication_text = ""
                            short_implication_text = ""
                            if summary_data and len(summary_data) == 6:
                                implication_text = summary_data[3] or ""       # 시사점
                                short_implication_text = summary_data[4] or ""  # 한줄 시사점
                            else:
                                implication_text = article.get("시사점", "") or ""
                                short_implication_text = article.get("한줄시사점", "") or ""

                            if implication_text:
                                st.markdown(implication_text)
                            if short_implication_text:
                                st.markdown(f"<span style='color:gray;font-style:italic;'>{short_implication_text}</span>", unsafe_allow_html=True)

                            st.markdown(
                                f"<span style='font-size:12px;color:#99a'>{article.get('날짜', '')} | {article.get('출처', '')}</span>",
                                unsafe_allow_html=True
                            )
                            if checked:
                                new_selection.append((major, minor, idx))

                            st.markdown("<div style='margin:0px;padding:0px;height:4px'></div>", unsafe_allow_html=True)

        st.session_state["important_selected_index"] = new_selection

        # 추가 / 삭제 / 교체 버튼 및 해당 기능 (기존 코드 유지)
        col_add, col_del, col_rep = st.columns([0.3, 0.35, 0.35])
        with col_add:
            if st.button("➕ 선택 기사 추가"):
                left_selected_keys = [k for k, v in st.session_state.article_checked_left.items() if v]
                if not left_selected_keys:
                    st.warning("왼쪽 뉴스검색 결과에서 적어도 1개 이상 선택해 주세요.")
                else:
                    added_count = 0
                    important = st.session_state.get("important_articles_preview", [])
                    for from_key in left_selected_keys:
                        m = re.match(r"^[^_]+_[0-9]+_(.+)$", from_key)
                        if not m:
                            continue
                        key_tail = m.group(1)
                        selected_article, article_link = None, None
                        for kw, arts in st.session_state.search_results.items():
                            for art in arts:
                                uid = re.sub(r'\W+', '', art['link'])[-16:]
                                if uid == key_tail:
                                    selected_article = art
                                    article_link = art["link"]
                                    break
                            if selected_article:
                                break
                        if not selected_article:
                            continue

                        keyword = extract_keyword_from_link(st.session_state.search_results, article_link)
                        cleaned_id = re.sub(r'\W+', '', selected_article['link'])[-16:]
                        sentiment = None
                        for k in st.session_state.keys():
                            if k.startswith("summary_") and cleaned_id in k:
                                sentiment = st.session_state[k][2]
                                break
                        if not sentiment:
                            _, _, sentiment, _, _ = summarize_article_from_url(
                                selected_article["link"], selected_article["title"]
                            )
                        new_article = {
                            "키워드": keyword,
                            "기사제목": selected_article["title"],
                            "감성": sentiment or "",
                            "링크": selected_article["link"],
                            "날짜": selected_article["date"],
                            "출처": selected_article["source"],
                            "시사점": ""  # 시사점 필드 초기값 빈 문자열
                        }
                        if not any(a["링크"] == new_article["링크"] for a in important):
                            important.append(new_article)
                            added_count += 1
                        st.session_state.article_checked_left[from_key] = False
                        st.session_state.article_checked[from_key] = False
                    st.session_state["important_articles_preview"] = important
                    if added_count > 0:
                        st.success(f"{added_count}건의 기사가 중요 기사 목록에 추가되었습니다.")
                    else:
                        st.info("추가된 새로운 기사가 없습니다.")
                    st.experimental_rerun()

        with col_del:
            if st.button("🗑 선택 기사 삭제"):
                important = st.session_state.get("important_articles_preview", [])
                remove_links = []
                for major, minor, idx in st.session_state["important_selected_index"]:
                    try:
                        link = major_map[major][minor][idx]["링크"]
                        remove_links.append(link)
                    except Exception:
                        continue
                important = [a for a in important if a.get("링크") not in remove_links]
                st.session_state["important_articles_preview"] = important
                st.session_state["important_selected_index"] = []
                st.experimental_rerun()

        with col_rep:
            if st.button("🔁 선택 기사 교체"):
                left_selected_keys = [k for k, v in st.session_state.article_checked_left.items() if v]
                right_selected_indexes = st.session_state["important_selected_index"]
                if len(left_selected_keys) != 1 or len(right_selected_indexes) != 1:
                    st.warning("왼쪽 1개, 오른쪽 1개만 선택해주세요.")
                    return
                from_key = left_selected_keys[0]
                (target_major, target_minor, target_idx) = right_selected_indexes[0]
                m = re.match(r"^[^_]+_[0-9]+_(.+)$", from_key)
                if not m:
                    st.warning("기사 식별자 파싱 실패")
                    return
                key_tail = m.group(1)
                selected_article, article_link = None, None
                for kw, art_list in st.session_state.search_results.items():
                    for art in art_list:
                        uid = re.sub(r'\W+', '', art['link'])[-16:]
                        if uid == key_tail:
                            selected_article = art
                            article_link = art["link"]
                            break
                    if selected_article:
                        break
                if not selected_article:
                    st.warning("왼쪽에서 선택한 기사 정보를 찾을 수 없습니다.")
                    return

                keyword = extract_keyword_from_link(st.session_state.search_results, article_link)
                cleaned_id = re.sub(r'\W+', '', selected_article['link'])[-16:]
                sentiment = None
                for k in st.session_state.keys():
                    if k.startswith("summary_") and cleaned_id in k:
                        sentiment = st.session_state[k][2]
                        break
                if not sentiment:
                    _, _, sentiment, _, _ = summarize_article_from_url(
                        selected_article["link"], selected_article["title"]
                    )
                important = st.session_state.get("important_articles_preview", [])
                remove_link = major_map[target_major][target_minor][target_idx]["링크"]
                important = [a for a in important if a.get("링크") != remove_link]
                new_article = {
                    "키워드": keyword,
                    "기사제목": selected_article["title"],
                    "감성": sentiment or "",
                    "링크": selected_article["link"],
                    "날짜": selected_article["date"],
                    "출처": selected_article["source"],
                    "시사점": ""  # 시사점 필드 초기값 빈 문자열
                }
                important.append(new_article)
                st.session_state["important_articles_preview"] = important
                st.session_state.article_checked_left[from_key] = False
                st.session_state.article_checked[from_key] = False
                st.session_state["important_selected_index"] = []
                st.success("중요 기사 교체 완료")
                st.experimental_rerun()

        st.markdown("---")
        st.markdown("📥 **리뷰한 중요 기사들을 엑셀로 다운로드하세요.**")
        articles_source = st.session_state.get("important_articles_preview", [])
        industry_keywords_all = []
        if st.session_state.get("use_industry_filter", False):
            for sublist in st.session_state.industry_major_sub_map.values():
                industry_keywords_all.extend(sublist)

        def enrich_article_for_excel(raw_article):
            link = raw_article.get("링크", "")
            keyword = raw_article.get("키워드", "")
            cleaned_id = re.sub(r"\W+", "", link)[-16:]

            one_line, summary, sentiment, implication, short_implication, full_text = None, None, None, None, None, None

            for k, v in st.session_state.items():
                if k.startswith("summary_") and cleaned_id in k and isinstance(v, tuple):
                    one_line, summary, sentiment, implication, short_implication, full_text = v
                    break

            if not sentiment:
                one_line, summary, sentiment, implication, short_implication, full_text = summarize_article_from_url(
                    link, raw_article.get("기사제목", "")
                )
            filter_hits = matched_filter_keywords(
                {"title": raw_article.get("기사제목", ""), "요약본": summary,
                 "요약": one_line, "full_text": full_text},
                ALL_COMMON_FILTER_KEYWORDS,
                industry_keywords_all
            )
            return {
                "키워드": keyword,
                "필터히트": ", ".join(filter_hits),
                "기사제목": safe_title(raw_article.get("기사제목", "")),
                "요약": one_line,
                "요약본": summary,
                "감성": sentiment,
                "시사점": implication,
                "한줄시사점": short_implication,   # 한줄 시사점 필드 추가
                "링크": link,
                "날짜": raw_article.get("날짜", ""),
                "출처": raw_article.get("출처", ""),
                "full_text": full_text or "",
            }
        summary_data = [enrich_article_for_excel(a) for a in articles_source]

        # 여기에서 엑셀 생성 시 한줄시사점 반영하여 통합
        def get_excel_with_joined_implications(summary_data, favorite_categories, excel_company_categories, search_results):
            import pandas as pd
            from io import BytesIO

            if not summary_data or len(summary_data) == 0:
                df_empty = pd.DataFrame(columns=["기업명", "표기명", "건수", "중요뉴스1", "중요뉴스2", "시사점"])
                output = BytesIO()
                with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                    df_empty.to_excel(writer, index=False, sheet_name='뉴스요약')
                    worksheet = writer.sheets['뉴스요약']
                    worksheet.set_column(0, 5, 30)
                output.seek(0)
                return output

            df = pd.DataFrame(summary_data)

            # 회사 리스트 (중복 제거 및 순서 유지)
            sector_list = []
            for cat in favorite_categories:
                sector_list.extend(favorite_categories[cat])
            sector_list = list(dict.fromkeys(sector_list))

            excel_sector_list = []
            for cat in excel_company_categories:
                excel_sector_list.extend(excel_company_categories[cat])
            excel_sector_list = list(dict.fromkeys(excel_sector_list))

            rows = []
            for idx, company in enumerate(sector_list):
                search_articles = search_results.get(company, [])

                filtered_articles = []
                for article in search_articles:
                    passes_common = any(kw in (article.get("title", "") + article.get("description", "")) for kw in ALL_COMMON_FILTER_KEYWORDS)
                    passes_industry = True
                    # 필요 시 산업별 필터링 로직 추가 가능

                    if passes_common and passes_industry:
                        filtered_articles.append(article)

                if st.session_state.get("remove_duplicate_articles", False):
                    filtered_articles = remove_duplicates(filtered_articles)

                total_count = len(filtered_articles)

                filtered_df = df[df.get("키워드", "") == company].sort_values(by='날짜', ascending=False)

                hl_news = ["", ""]
                implications = ["", ""]
                short_imps = ["", ""]

                for i, art in enumerate(filtered_df.itertuples()):
                    if i > 1:
                        break
                    date_val = getattr(art, "날짜", "") or ""
                    title_val = getattr(art, "기사제목", "") or getattr(art, "제목", "")
                    link_val = getattr(art, "링크", "") or getattr(art, "link", "")
                    short_imp_val = getattr(art, "한줄시사점", "") or ""

                    display_text = f"({clean_excel_formula_text(date_val)}){clean_excel_formula_text(title_val)}"
                    if title_val and link_val:
                        hl_news[i] = f'=HYPERLINK("{clean_excel_formula_text(link_val)}", "{display_text}")'
                    else:
                        hl_news[i] = display_text or ""

                    implications[i] = getattr(art, "시사점", "") or ""
                    short_imps[i] = short_imp_val

                # 시사점 및 한줄시사점 번호 붙여서 병합
                merged_implications = ""
                for n in range(2):
                    if implications[n]:
                        merged_implications += f"{n+1}. {implications[n]}\n"
                for n in range(2):
                    if short_imps[n]:
                        merged_implications += f"{n+1}. {short_imps[n]}\n"

                rows.append({
                    "기업명": company,
                    "표기명": excel_sector_list[idx] if idx < len(excel_sector_list) else "",
                    "건수": total_count,
                    "중요뉴스1": hl_news[0],
                    "중요뉴스2": hl_news[1],
                    "시사점": merged_implications.strip(),
                })

            result_df = pd.DataFrame(rows, columns=["기업명", "표기명", "건수", "중요뉴스1", "중요뉴스2", "시사점"])

            output = BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                result_df.to_excel(writer, index=False, sheet_name='뉴스요약')
                worksheet = writer.sheets['뉴스요약']
                for i, col in enumerate(result_df.columns):
                    worksheet.set_column(i, i, 30)
            output.seek(0)
            return output

        excel_data = get_excel_with_joined_implications(summary_data, favorite_categories, excel_company_categories, st.session_state.search_results)

        st.download_button(
            label="📥 중요 기사 최종 엑셀 다운로드 (맞춤 양식)",
            data=excel_data.getvalue(),
            file_name=f"중요뉴스_최종선정_양식_{datetime.now().strftime('%Y%m%d')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

if st.session_state.get("search_results"):
    filtered_results = {}
    for keyword, articles in st.session_state["search_results"].items():
        filtered_articles = [a for a in articles if article_passes_all_filters(a)]

        if st.session_state.get("remove_duplicate_articles", False):
            filtered_articles = remove_duplicates(filtered_articles)

        if filtered_articles:
            filtered_results[keyword] = filtered_articles

    render_articles_with_single_summary_and_telegram(
        filtered_results,
        st.session_state.show_limit,
        show_sentiment_badge=st.session_state.get("show_sentiment_badge", False),
        enable_summary=st.session_state.get("enable_summary", True)
    )

    selected_companies = []
    for cat in st.session_state.get("cat_multi", []):
        selected_companies.extend(favorite_categories.get(cat, []))
    selected_companies = list(set(selected_companies))

    # kiscd_map과 cmpCD_map 모두에서 회사명에 매칭되는 키 값 가져오기
    kiscd_filtered = {c: kiscd_map[c] for c in selected_companies if c in kiscd_map}
    cmpcd_filtered = {c: config.get("cmpCD_map", {}).get(c, "") for c in selected_companies}

    # 두 맵을 합치는 함수 (kiscd_filtered 기본에 cmpcd_filtered도 합칠 수 있도록)
    # fetch_and_display_reports가 kiscd만 받으므로 확장 필요
    # 여기서는 kiscd_filtered 넘기고, fetch_and_display_reports 내부에서 cmpCD_map 참조 권장

    fetch_and_display_reports(kiscd_filtered)

else:
    st.info("뉴스 검색 결과가 없습니다. 먼저 검색을 실행해 주세요.")
