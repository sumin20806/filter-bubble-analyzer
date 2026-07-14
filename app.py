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
import yt_dlp
from google import genai

# ---------------------------------------------------------
# 기본 설정
# ---------------------------------------------------------
st.set_page_config(page_title="필터 버블 분석기", page_icon="🎯", layout="centered")

plt.rcParams['font.family'] = 'NanumGothic'
plt.rcParams['axes.unicode_minus'] = False

CATEGORY_LIST = [
    "정치/시사", "경제/재테크", "게임", "음악", "먹방/요리",
    "뷰티/패션", "스포츠", "교육/강의", "IT/과학기술",
    "브이로그/일상", "코미디/개그", "여행", "동물/펫",
    "영화/드라마 리뷰", "기타",
]

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
    prompt = f"""다음 유튜브 영상을 아래 카테고리 목록 중 정확히 하나로만 분류해줘.
반드시 목록에 있는 단어 그대로, 다른 설명 없이 카테고리 이름만 출력해.

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
# 사이드바: 설정
# ---------------------------------------------------------
st.sidebar.header("⚙️ 설정")
gemini_api_key = st.sidebar.text_input("Gemini API 키", type="password", help="https://aistudio.google.com/apikey 에서 발급")
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
