# =========================================================
# 필터 버블 분석 웹앱 (Streamlit)
# 실행: streamlit run app.py
# =========================================================

import os
import re
import math
import time
from collections import Counter

import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import yt_dlp
from google import genai

# ---------------------------------------------------------
# 기본 설정
# ---------------------------------------------------------
st.set_page_config(page_title="필터 버블 분석기", page_icon="🎯", layout="centered")

# 한글 폰트 등록 (packages.txt로 설치된 나눔고딕을 matplotlib에 인식시킴)
_NANUM_PATH = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"
if os.path.exists(_NANUM_PATH):
    fm.fontManager.addfont(_NANUM_PATH)
    plt.rcParams['font.family'] = 'NanumGothic'
else:
    st.warning("한글 폰트를 찾지 못해 그래프 글자가 깨질 수 있습니다. (packages.txt 확인 필요)")
plt.rcParams['axes.unicode_minus'] = False

CATEGORY_DEFINITIONS = {
    "정치/시사": "정치인, 정당, 선거, 정부 정책, 국회, 시사 뉴스, 사회적 이슈를 다루는 영상",
    "경제/재테크": "주식, 부동산, 코인, 재테크 방법, 경제 지표, 기업 실적, 금융 상품을 다루는 영상",
    "게임": "게임 플레이, 공략, 리뷰, e스포츠 중계 등 게임 관련 콘텐츠",
    "음악": "노래, 뮤직비디오, 커버, 라이브 공연, 앨범 리뷰, 음악 방송",
    "먹방/요리": "음식 먹는 콘텐츠(먹방), 레시피, 요리 과정, 맛집 소개",
    "뷰티/패션": "화장품, 메이크업 튜토리얼, 스타일링, 패션 아이템 리뷰",
    "스포츠": "스포츠 경기 중계, 하이라이트, 운동선수 관련, 스포츠 분석/해설",
    "교육/강의": "학습 콘텐츠, 강의, 시험 준비, 지식 전달 목적의 튜토리얼",
    "IT/과학기술": "신제품 리뷰, 소프트웨어/앱 사용법, 과학 실험, 기술 트렌드",
    "브이로그/일상": "일상 기록, 개인 브이로그, 특정 주제 없이 생활을 보여주는 영상",
    "코미디/개그": "웃음을 목적으로 한 콩트, 몰래카메라, 개그 콘텐츠",
    "여행": "국내외 여행 기록, 여행지 소개, 여행 팁",
    "동물/펫": "반려동물, 야생동물, 동물 행동/훈련 관련 영상",
    "영화/드라마 리뷰": "영화·드라마 리뷰, 줄거리 요약, 결말 포함 해설, 비평",
    "기타": "위 14개 카테고리 중 어디에도 명확히 속하지 않는 영상",
}
CATEGORY_LIST = list(CATEGORY_DEFINITIONS.keys())

GEMINI_MODEL = "gemini-3.5-flash"


# ---------------------------------------------------------
# 분석 함수들
# ---------------------------------------------------------
def is_valid_youtube_url(url):
    patterns = [r"(?:v=)([0-9A-Za-z_-]{11})", r"youtu\.be/([0-9A-Za-z_-]{11})", r"shorts/([0-9A-Za-z_-]{11})"]
    return any(re.search(p, url) for p in patterns)


def get_video_info(url, cookie_path=None):
    ydl_opts = {
        "quiet": True,
        "skip_download": True,
        "no_warnings": True,
        "extractor_args": {"youtube": {"player_client": ["android"]}},
    }
    if cookie_path and os.path.exists(cookie_path):
        ydl_opts["cookiefile"] = cookie_path
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
        return {"title": info.get("title", ""), "description": (info.get("description") or "")[:300]}
    except Exception as e:
        msg = str(e)
        if "Sign in to confirm" in msg:
            return {"error": "봇 차단 (쿠키 필요)"}
        elif "not available in your country" in msg:
            return {"error": "국가 제한 영상"}
        else:
            return {"error": "정보 조회 실패"}


def classify_category(client, title, description):
    definitions_text = "\n".join(f"- {name}: {desc}" for name, desc in CATEGORY_DEFINITIONS.items())
    prompt = f"""다음 유튜브 영상을 아래 카테고리 목록 중 정확히 하나로만 분류해줘.
반드시 목록에 있는 단어 그대로, 다른 설명 없이 카테고리 이름만 출력해.

[카테고리 정의]
{definitions_text}

[분류 규칙]
- 영상을 만든/등장하는 사람의 정체성(예: 연예인, 유튜버)이 아니라, 영상이 실제로 다루는 "발언·행동의 주제"를 기준으로 분류해.
  예: 연예인이 등장해도 내용이 특정 정당 지지 발언이면 "정치/시사"로 분류.
- 여러 카테고리에 걸쳐 있으면, 영상 제목/설명에서 가장 비중 있게 다뤄지는 주제 하나를 선택해.
- 어떤 카테고리 정의에도 명확히 맞지 않으면 "기타"를 선택해.

카테고리 목록: {', '.join(CATEGORY_LIST)}

영상 제목: {title}
영상 설명: {description}

카테고리:"""
    try:
        response = client.interactions.create(model=GEMINI_MODEL, input=prompt)
        result = response.output_text.strip()
        if result not in CATEGORY_LIST:
            for c in CATEGORY_LIST:
                if c in result:
                    return c
            return "기타"
        return result
    except Exception:
        return "기타"


def calculate_diversity(category_list, all_categories):
    unique_categories = set(category_list)
    return len(unique_categories) / len(all_categories), unique_categories


def calculate_shannon_entropy(category_list):
    counter = Counter(category_list)
    total = len(category_list)
    entropy = 0.0
    for count in counter.values():
        p = count / total
        entropy -= p * math.log2(p)
    return entropy, counter


def calculate_bias_percent(entropy, num_unique_categories):
    if num_unique_categories <= 1:
        return 100.0
    max_entropy = math.log2(num_unique_categories)
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
    return (1 - normalized_entropy) * 100


# ---------------------------------------------------------
# 시각화 함수 (Streamlit용: fig를 반환)
# ---------------------------------------------------------
def fig_distribution(counter):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(list(counter.keys()), list(counter.values()), color='skyblue')
    ax.set_xlabel("카테고리")
    ax.set_ylabel("영상 수")
    ax.set_title("시청 영상 카테고리 분포")
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    fig.tight_layout()
    return fig


def fig_bias_gauge(bias_percent, threshold):
    fig, ax = plt.subplots(figsize=(8, 2))
    ax.barh(0, 40, color="#8bd3a0", left=0)
    ax.barh(0, 30, color="#f6d060", left=40)
    ax.barh(0, 30, color="#f08a8a", left=70)
    ax.barh(0, bias_percent, color="#333333", height=0.25)
    ax.text(bias_percent, 0.55, f"{bias_percent:.1f}%", ha="center", va="bottom", fontsize=13, fontweight="bold")
    ax.axvline(threshold, color="red", linestyle="--", linewidth=1.5)
    ax.text(threshold, -0.55, f"경고 기준 {threshold:.0f}%", ha="center", va="top", color="red", fontsize=10)
    ax.set_xlim(0, 100)
    ax.set_ylim(-1, 1)
    ax.set_yticks([])
    ax.set_xlabel("편향도 (%)")
    ax.set_title("편향도 게이지")
    fig.tight_layout()
    return fig


def fig_entropy_comparison(entropy, num_unique_categories):
    max_entropy = math.log2(num_unique_categories) if num_unique_categories > 1 else 0.0
    labels = ["실제 엔트로피 H(X)", "최대 엔트로피 H_max"]
    values = [entropy, max_entropy]
    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(labels, values, color=["#5b8def", "#cccccc"])
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.03, f"{v:.2f} bit", ha="center", va="bottom", fontsize=11)
    ax.set_ylabel("엔트로피 (bit)")
    ax.set_title("샤논 엔트로피: 실제 vs 최대")
    ax.set_ylim(0, max(values) * 1.3 if max(values) > 0 else 1)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------
# API 키 발급 튜토리얼 (팝업 창)
# ---------------------------------------------------------
@st.dialog("🔑 Gemini API 키 발급 방법")
def show_api_key_tutorial():
    st.markdown("""
**Gemini API 키**는 이 앱이 영상을 자동으로 분류할 때 사용하는 열쇠예요.
아래 순서대로 따라 하면 5분 안에 무료로 발급받을 수 있어요.

---

**1단계. Google AI Studio 접속**
👉 [aistudio.google.com/apikey](https://aistudio.google.com/apikey) 로 이동하세요.

**2단계. 구글 계정으로 로그인**
평소 쓰는 구글 계정(지메일)으로 로그인하면 돼요. 별도 가입 절차 없어요.

**3단계. "Create API key" 버튼 클릭**
화면에 보이는 **Create API key** (또는 **API 키 만들기**) 버튼을 누르세요.
새 프로젝트를 만들지, 기존 프로젝트를 쓸지 물어보면 아무거나 선택해도 괜찮아요.

**4단계. 생성된 키 복사**
`AIzaSy`로 시작하는 긴 문자열이 생성돼요. 옆에 있는 복사 아이콘을 눌러 복사하세요.

**5단계. 이 앱에 붙여넣기**
왼쪽 사이드바의 **"Gemini API 키"** 입력칸에 붙여넣으면 끝!
""")
    st.warning("⚠️ API 키는 비밀번호와 같아요. 캡처해서 SNS에 올리거나 남에게 공유하지 마세요. 무료 사용량에는 한도가 있으니, 발표/시연이 끝나면 AI Studio에서 키를 삭제하는 걸 추천해요.")
    if st.button("닫기", use_container_width=True):
        st.rerun()


# ---------------------------------------------------------
# 사이드바: 설정
# ---------------------------------------------------------
st.sidebar.header("⚙️ 설정")

key_col, help_col = st.sidebar.columns([5, 1])
with key_col:
    gemini_api_key = st.text_input("Gemini API 키", type="password", help="https://aistudio.google.com/apikey 에서 발급")
with help_col:
    st.write("")
    st.write("")
    if st.button("➕", key="api_key_help_btn", help="API 키 발급 방법 보기"):
        show_api_key_tutorial()

bias_threshold = st.sidebar.slider("편향도 경고 기준 (%)", min_value=0, max_value=100, value=70, step=5)
cookie_file = st.sidebar.file_uploader("유튜브 쿠키 파일 (선택, cookies.txt)", type=["txt"])

cookie_path = None
if cookie_file is not None:
    cookie_path = "/tmp/cookies.txt"
    with open(cookie_path, "wb") as f:
        f.write(cookie_file.getbuffer())

# ---------------------------------------------------------
# 메인 화면
# ---------------------------------------------------------
st.title("🎯 유튜브 필터 버블 분석기")
st.write("내가 시청한 유튜브 영상 링크를 입력하면, 카테고리 다양성 지수와 샤논 엔트로피 기반 편향도를 계산해줘요.")

urls_text = st.text_area(
    "유튜브 링크 (한 줄에 하나씩 입력)",
    height=180,
    placeholder="https://www.youtube.com/watch?v=xxxxxxxxxxx\nhttps://youtu.be/xxxxxxxxxxx\n...",
)

analyze_btn = st.button("🔍 분석 시작", type="primary", use_container_width=True)

if analyze_btn:
    if not gemini_api_key:
        st.error("사이드바에 Gemini API 키를 먼저 입력해주세요.")
        st.stop()

    raw_urls = [u.strip() for u in urls_text.splitlines() if u.strip()]
    urls = [u for u in raw_urls if is_valid_youtube_url(u)]
    invalid_count = len(raw_urls) - len(urls)

    if invalid_count > 0:
        st.warning(f"유효하지 않은 링크 {invalid_count}개는 제외하고 분석합니다.")

    if not urls:
        st.error("분석할 유효한 유튜브 링크가 없습니다.")
        st.stop()

    client = genai.Client(api_key=gemini_api_key)

    categories = []
    progress = st.progress(0, text="영상 분석 중...")
    log_box = st.empty()
    logs = []

    for i, url in enumerate(urls, 1):
        info = get_video_info(url, cookie_path)
        if info is None or "error" in info:
            logs.append(f"⚠️ 건너뜀 ({info.get('error', '알 수 없는 오류') if info else '오류'}): {url}")
        else:
            cat = classify_category(client, info["title"], info["description"])
            categories.append(cat)
            logs.append(f"✅ {info['title'][:40]}...  →  [{cat}]")
            time.sleep(1.0)  # API 요청 속도 제한 방지
        progress.progress(i / len(urls), text=f"영상 분석 중... ({i}/{len(urls)})")
        log_box.text("\n".join(logs[-8:]))  # 최근 로그만 표시

    progress.empty()

    if not categories:
        st.error("분석 가능한 영상이 없습니다.")
        st.stop()

    diversity, unique_cats = calculate_diversity(categories, CATEGORY_LIST)
    entropy, counter = calculate_shannon_entropy(categories)
    bias_percent = calculate_bias_percent(entropy, len(unique_cats))

    st.divider()
    st.subheader("📊 최종 분석 결과")

    col1, col2, col3 = st.columns(3)
    col1.metric("분석된 영상 수", f"{len(categories)}개")
    col2.metric("다양성 지수", f"{diversity:.3f}", help=f"{len(unique_cats)}/{len(CATEGORY_LIST)} 카테고리 사용")
    col3.metric("편향도", f"{bias_percent:.1f}%")

    if bias_percent >= bias_threshold:
        st.error(f"🚨 경고: 편향도가 {bias_threshold}% 이상입니다! 특정 카테고리에 시청 콘텐츠가 쏠려있어 필터 버블 상태일 가능성이 높습니다.")
    else:
        st.success(f"✅ 편향도가 {bias_threshold}% 미만으로, 비교적 다양한 콘텐츠를 시청하고 있습니다.")

    st.pyplot(fig_distribution(counter))
    st.pyplot(fig_bias_gauge(bias_percent, bias_threshold))
    st.pyplot(fig_entropy_comparison(entropy, len(unique_cats)))
