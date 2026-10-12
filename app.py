import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import json
import ast
import re
from datetime import datetime

from utils.preprocessing import (
    clean_destinations,
    clean_hotels,
    clean_restaurants
)
from utils.kmeans import run_kmeans, build_feature_matrix
from utils.topsis import rank_by_suitability, explain_top_criteria, recommended_weights
from utils.learning import build_name_index, community_scores, learn_weights, MIN_LABELS, LEARN_FEATURES
from utils.evaluation import offline_evaluation, online_metrics, clustering_report


from utils.feedback_manager import (
    load_feedback_data,
    log_implicit_view,
    save_explicit_feedback,
    get_feedback_stats,
    reset_feedback_data,
    is_cloud_configured,
    sync_local_to_cloud,
)
from sklearn.metrics import silhouette_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Hệ thống Trợ giúp Ra quyết định Du lịch Việt Nam",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# =========================================================
# CUSTOM CSS (CLEAN INTER DESIGN SYSTEM)
# =========================================================
st.markdown("""
<style>

:root {
    --primary:      #8C2D19; /* Nâu đỏ sang trọng */
    --primary-dk:   #6E2010;
    --primary-lt:   #FAF1ED;
    --primary-glow: rgba(140, 45, 25, 0.12);
    --bg-base:      #FAF8F5; /* Nền kem ấm cổ điển */
    --bg-card:      #FFFFFF;
    --cream-dk:     #EFEBE2;
    --text-main:    #2A2421;
    --text-muted:   #6D645E;
    --text-2:       #483F3A;
    --white:        #FFFFFF;
    --border:       #E8E0D7;
    --border-light: #F4EFEB;
    --gold:         #B87C28;
    --gold-bg:      #FCF8EF;
    --shadow-sm:    0 1px 4px rgba(40, 20, 15, 0.05);
    --shadow-md:    0 4px 14px rgba(40, 20, 15, 0.07);
    --shadow-lg:    0 10px 28px rgba(40, 20, 15, 0.10);
    --r-sm:         6px;
    --r-md:         10px;
    --font:         'Times New Roman', Times, serif;
}

/* ═══ BASE FONT & PAGE BACKGROUND ═══ */
html, body, .stApp {
    font-family: var(--font) !important;
    background-color: var(--bg-base) !important;
    color: var(--text-main);
    font-size: 15px;
}

/* Hide Streamlit default chrome */
#MainMenu, footer, [data-testid="stDecoration"] { visibility: hidden; }

/* ═══ ĐỒNG BỘ CỠ CHỮ THÔNG THƯỜNG (KHÔNG IN ĐẬM TRÀN LAN) ═══ */
p, li, .stMarkdown p {
    font-family: var(--font) !important;
    font-size: 0.95rem;
    font-weight: 400;
    color: var(--text-main);
    line-height: 1.55;
}

/* ═══ QUY TẮC BẮT BUỘC: NỀN ĐỎ / NÂU ĐỎ THÌ CHỮ PHẢI TRẮNG 100% ═══ */

/* 1. Nút bấm chính (Primary Button) - Toàn bộ nút và mọi phần tử con đều chữ trắng */
button[kind="primary"],
button[data-testid="baseButton-primary"],
.stButton > button[kind="primary"],
.stFormSubmitButton > button[kind="primary"],
.stFormSubmitButton > button[data-testid="baseButton-primary"],
.stDownloadButton > button[kind="primary"] {
    background-color: var(--primary) !important;
    background: var(--primary) !important;
    color: #FFFFFF !important;
    border: 1px solid var(--primary-dk) !important;
    border-radius: var(--r-sm) !important;
    padding: 8px 18px !important;
    font-size: 0.95rem !important;
    font-weight: 700 !important;
    box-shadow: 0 2px 6px rgba(140, 45, 25, 0.25) !important;
    transition: all 0.2s ease !important;
}

button[kind="primary"] *,
button[kind="primary"] p,
button[kind="primary"] span,
button[kind="primary"] div,
button[data-testid="baseButton-primary"] *,
button[data-testid="baseButton-primary"] p,
button[data-testid="baseButton-primary"] span,
button[data-testid="baseButton-primary"] div,
.stButton > button[kind="primary"] *,
.stButton > button[kind="primary"] p,
.stButton > button[kind="primary"] span,
.stButton > button[kind="primary"] div,
.stFormSubmitButton > button[kind="primary"] *,
.stFormSubmitButton > button[kind="primary"] p,
.stFormSubmitButton > button[kind="primary"] span,
.stFormSubmitButton > button[kind="primary"] div,
.stDownloadButton > button[kind="primary"] *,
.stDownloadButton > button[kind="primary"] p,
.stDownloadButton > button[kind="primary"] span,
.stDownloadButton > button[kind="primary"] div {
    color: #FFFFFF !important;
    fill: #FFFFFF !important;
    font-family: var(--font) !important;
    font-size: 0.95rem !important;
    font-weight: 700 !important;
}

/* Hover nút đỏ */
button[kind="primary"]:hover,
button[data-testid="baseButton-primary"]:hover,
.stButton > button[kind="primary"]:hover {
    background-color: var(--primary-dk) !important;
    background: var(--primary-dk) !important;
    border-color: #551608 !important;
    color: #FFFFFF !important;
}
button[kind="primary"]:hover *,
button[data-testid="baseButton-primary"]:hover *,
.stButton > button[kind="primary"]:hover * {
    color: #FFFFFF !important;
}

/* 2. Banner Hero (Nền nâu đỏ đậm) */
.hero {
    background: linear-gradient(135deg, #50170B 0%, #300C06 100%) !important;
    border-radius: var(--r-md);
    padding: 26px 20px;
    text-align: center;
    margin-bottom: 22px;
    box-shadow: var(--shadow-md);
    border: 1px solid rgba(140, 45, 25, 0.4);
}
.hero, .hero * {
    font-family: var(--font) !important;
}
.hero-title, .hero h1, .hero h2 {
    font-size: 1.85rem !important; /* Cỡ to */
    font-weight: 800 !important; /* In đậm */
    text-transform: uppercase !important; /* In hoa */
    letter-spacing: 0.04em !important;
    line-height: 1.35 !important;
    color: #FFFFFF !important; /* Chữ trắng tuyệt đối */
    margin-bottom: 8px !important;
}
.hero-description, .hero p {
    color: #F5EAE4 !important; /* Trắng sáng */
    font-size: 0.92rem !important;
    max-width: 660px;
    margin: 0 auto;
    line-height: 1.55;
    font-weight: 400 !important;
    font-style: normal !important;
}
.hero-eyebrow {
    color: #F8D0A0 !important; /* Vàng kim nhạt sang trọng */
    font-size: 0.80rem !important;
    font-weight: 600 !important;
    text-transform: uppercase;
    letter-spacing: 0.10em;
    margin-bottom: 6px;
    display: inline-block;
}

/* 3. Huy hiệu Match-Badge (Nền nâu đỏ -> Chữ trắng nổi bật) */
.match-badge {
    background-color: var(--primary) !important;
    background: var(--primary) !important;
    color: #FFFFFF !important;
    border: none !important;
    padding: 2px 8px;
    border-radius: 4px;
    font-weight: 700;
    font-size: 0.80rem;
    font-family: var(--font);
    display: inline-flex;
    align-items: center;
    box-shadow: 0 1px 3px rgba(140, 45, 25, 0.25);
}
.match-badge * {
    color: #FFFFFF !important;
}

/* ═══ NÚT PHỤ (SECONDARY) - IN ĐẬM CHỮ ═══ */
button[kind="secondary"],
button[data-testid="baseButton-secondary"],
.stButton > button,
.stButton > button[kind="secondary"],
.stFormSubmitButton > button[kind="secondary"],
.stDownloadButton > button[kind="secondary"] {
    background-color: #FFFFFF !important;
    color: var(--text-main) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--r-sm) !important;
    padding: 7px 18px !important;
    box-shadow: var(--shadow-sm) !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
    transition: all 0.2s ease !important;
}
button[kind="secondary"] *,
button[data-testid="baseButton-secondary"] *,
.stButton > button *,
.stButton > button[kind="secondary"] *,
.stFormSubmitButton > button[kind="secondary"] *,
.stDownloadButton > button[kind="secondary"] * {
    color: var(--text-main) !important;
    font-family: var(--font) !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
}
button[kind="secondary"]:hover,
button[data-testid="baseButton-secondary"]:hover {
    border-color: var(--primary) !important;
    background-color: var(--primary-lt) !important;
    color: var(--primary) !important;
}
button[kind="secondary"]:hover * {
    color: var(--primary) !important;
}

/* Align button right in header column */
div[data-testid="column"]:nth-of-type(2) .stButton {
    display: flex;
    justify-content: flex-end;
}

/* ═══ TIÊU ĐỀ MỤC & PHÂN CẤP (CHỈ TO VÀ ĐẬM Ở MỤC CẦN THIẾT) ═══ */
h1, h2, h3, h4, .admin-table-title {
    font-family: var(--font) !important;
    font-weight: 700 !important;
    color: var(--primary) !important;
    line-height: 1.35 !important;
}

.section-title {
    font-family: var(--font) !important;
    font-size: 1.15rem !important;
    font-weight: 700 !important;
    color: var(--primary) !important;
    margin-bottom: 14px !important;
    border-bottom: 1.5px solid var(--border);
    padding-bottom: 6px;
    line-height: 1.35 !important;
}

/* Card Containers */
div[data-testid="stVerticalBlockBorderWrapper"] {
    background: var(--bg-card);
    border: 1px solid var(--border) !important;
    border-radius: var(--r-md) !important;
    box-shadow: var(--shadow-sm) !important;
    margin-bottom: 20px;
}
div[data-testid="stVerticalBlockBorderWrapper"] > div {
    padding: 20px 24px !important;
}

/* ═══ FORM LABELS & INPUTS ═══ */
.stTextInput label, .stNumberInput label, .stSlider label, .stTextArea label, .stSelectbox label {
    font-family: var(--font) !important;
    font-size: 0.90rem !important;
    font-weight: 600 !important;
    color: var(--text-2) !important;
    letter-spacing: normal !important;
    text-transform: none !important;
}
.stTextInput input, .stTextArea textarea, .stNumberInput input, .stSelectbox select {
    font-family: var(--font) !important;
    font-size: 0.95rem !important;
    font-weight: 400 !important;
    color: var(--text-main) !important;
    background-color: #FFFFFF !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--r-sm) !important;
    padding: 8px 12px !important;
}
.stTextInput input::placeholder, .stTextArea textarea::placeholder {
    font-family: var(--font) !important;
    font-style: italic !important;
    color: #9C938D !important;
    font-size: 0.90rem !important;
}
.stTextInput input:focus, .stTextArea textarea:focus, .stNumberInput input:focus {
    border-color: var(--primary) !important;
    box-shadow: 0 0 0 2px var(--primary-glow) !important;
}

/* Chú thích & Lựa chọn Radio */
.stCaption, [data-testid="stCaptionContainer"] {
    font-family: var(--font) !important;
    font-size: 0.82rem !important;
    color: var(--text-muted) !important;
    font-weight: 400 !important;
}
.stRadio label, .stRadio div[role="radiogroup"] label {
    font-family: var(--font) !important;
    font-size: 0.92rem !important;
    font-weight: 400 !important;
    color: var(--text-main) !important;
}

/* Tabs */
button[data-baseweb="tab"] {
    font-family: var(--font) !important;
    font-size: 0.95rem !important;
    font-weight: 500 !important;
    color: var(--text-muted) !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
    font-weight: 700 !important;
    color: var(--primary) !important;
}

/* ═══ KPI METRIC CARDS (ADMIN) ═══ */
.kpi-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--r-md);
    padding: 16px 14px;
    text-align: center;
    box-shadow: var(--shadow-sm);
    border-top: 3px solid var(--primary);
}
.kpi-value {
    color: var(--primary);
    font-size: 1.75rem;
    font-weight: 700;
    font-family: var(--font);
    line-height: 1.25;
}
.kpi-label {
    color: var(--text-muted);
    font-size: 0.74rem;
    font-weight: 500;
    letter-spacing: 0.04em;
    margin-top: 6px;
    font-family: var(--font);
}

/* ═══ DESTINATION ITEM CARDS ═══ */
.dest-card-img {
    width: 100%; height: 165px; object-fit: cover;
    border-radius: var(--r-sm);
    box-shadow: var(--shadow-sm);
}
.dest-name {
    font-family: var(--font) !important;
    font-size: 1.12rem !important;
    font-weight: 700 !important;
    color: var(--primary) !important;
    line-height: 1.35;
    margin-bottom: 4px;
}
.cluster-tag {
    font-size: 0.78rem; font-weight: 500; padding: 2px 8px;
    border-radius: 4px; display: inline-block;
    background: #F4EFEB; color: var(--text-2);
    border: 1px solid var(--border);
    font-family: var(--font);
}
.dest-meta {
    display: flex; align-items: center; gap: 8px; flex-wrap: wrap;
    font-size: 0.88rem; color: var(--text-2); margin-bottom: 6px;
    line-height: 1.4;
    font-family: var(--font);
}
.meta-sep {
    color: #D6CBC2; margin: 0 2px;
}

/* Custom Xem Thêm Details/Summary */
details.desc-details > summary {
    list-style: none;
    outline: none;
    cursor: pointer;
}
details.desc-details > summary::-webkit-details-marker {
    display: none;
}
details.desc-details .show-less {
    display: none;
}
details.desc-details[open] .short-text {
    display: none !important;
}
details.desc-details[open] .show-less {
    display: inline !important;
}

</style>

""", unsafe_allow_html=True)


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def clean_title_icon(text):
    if not text or pd.isna(text):
        return ""
    return re.sub(r'[\U00010000-\U0010ffff\u2600-\u27ff\u2300-\u23ff\ufe0f]', '', str(text)).strip()


def safe_image_tag(url, class_name="dest-card-img"):
    url_str = str(url).strip() if (url and not pd.isna(url)) else ""
    if url_str and (url_str.startswith("http://") or url_str.startswith("https://")):
        return f'<img src="{url_str}" class="{class_name}" alt="Destination" onerror="this.onerror=null; this.src=\'https://images.unsplash.com/photo-1528127269322-539801943592?auto=format&fit=crop&w=800&q=80\';"/>'
    return f'<div class="{class_name}" style="background:#F3F4F6; display:flex; align-items:center; justify-content:center; color:#9CA3AF; font-size:0.85rem; min-height:160px; border:1px solid #E5E7EB;">Hình ảnh điểm đến</div>'


def format_price_range(min_val, max_val, unit="VNĐ"):
    if pd.isna(min_val) and pd.isna(max_val):
        return "Miễn phí / Tùy chọn"
    if min_val == 0 and max_val == 0:
        return "Miễn phí"
    if pd.isna(min_val) or min_val == 0:
        return f"Khoảng {max_val:,.0f} {unit}"
    if pd.isna(max_val) or min_val == max_val:
        return f"{min_val:,.0f} {unit}"
    return f"{min_val:,.0f} - {max_val:,.0f} {unit}"


CLUSTER_COLOR_MAP = {
    0: "#2B507C",
    1: "#C98A2E",
    2: "#2A7B5C",
    3: "#8A3B2A",
    4: "#5B3A7B",
}


@st.cache_data
def load_data():
    dest_df = clean_destinations()
    hotel_df = clean_hotels()
    rest_df = clean_restaurants()
    return dest_df, hotel_df, rest_df


@st.cache_data
def load_clustered(_dest_df, k=None):
    result_df, _ = run_kmeans(_dest_df, k=k)
    return result_df







def extract_intent_keywords(text: str):
    if not text or not text.strip():
        return [], {}

    text_lower = text.lower()
    boosts = {"nature": 0, "history": 0, "resort": 0, "budget": 0}

    nature_kws = ["thiên nhiên", "xanh mát", "phong cảnh", "núi", "rừng", "hồ", "thác", "suối", "cảnh đẹp", "cây xanh", "sinh thái"]
    history_kws = ["lịch sử", "văn hóa", "di tích", "chùa", "đền", "cổ kính", "lâu đời", "bảo tàng", "phố cổ"]
    resort_kws = ["nghỉ dưỡng", "thư giãn", "spa", "resort", "sang trọng", "yên tĩnh", "tận hưởng"]
    beach_kws = ["biển", "đảo", "vịnh", "sóng", "cát trắng", "lặn ngắm san hô"]

    found_kws = []
    for kw in nature_kws + beach_kws:
        if kw in text_lower:
            found_kws.append(kw)
            boosts["nature"] += 1
    for kw in history_kws:
        if kw in text_lower:
            found_kws.append(kw)
            boosts["history"] += 1
    for kw in resort_kws:
        if kw in text_lower:
            found_kws.append(kw)
            boosts["resort"] += 1

    return list(set(found_kws)), boosts


@st.cache_resource
def load_search_model(_dest_df):
    """Ma trận đặc trưng K-Means + mô hình TF-IDF (tính 1 lần, không tính lại mỗi lần rerun)."""
    return build_feature_matrix(_dest_df)


@st.cache_data(show_spinner="Đang chạy đánh giá offline trên các kịch bản mô phỏng...")
def run_offline_eval(_result_df, _vectorizer, _tfidf_matrix, community_key, _hotel_df=None, _rest_df=None):
    return offline_evaluation(
        _result_df, _vectorizer, _tfidf_matrix, community=dict(community_key),
        hotel_df=_hotel_df, rest_df=_rest_df
    )


# Load Data
dest_df, hotel_df, rest_df = load_data()
result_df = load_clustered(dest_df, k=None)
feature_matrix, vectorizer, scaler, _, tfidf_matrix = load_search_model(dest_df)


# =========================================================
# HEADER BAR WITH ADMIN LOGIN BUTTON AT TOP RIGHT
# =========================================================
if "is_admin_logged_in" not in st.session_state:
    st.session_state["is_admin_logged_in"] = False
if "active_view" not in st.session_state:
    st.session_state["active_view"] = "SURVEY"

h_col1, h_col2 = st.columns([3, 1])

with h_col1:
    st.markdown("""
    <div style="padding: 4px 0;">
        <div style="font-family:var(--font); font-size:1.35rem; font-weight:700; color:var(--primary); line-height:1.35;">
            Hệ thống Trợ giúp Ra quyết định Du lịch
        </div>
        <div style="font-size:0.76rem; font-weight:500; color:var(--text-muted); margin-top:2px; letter-spacing:0.04em;">
            ĐỒ ÁN DSS · K-MEANS · TOPSIS · CONTENT-BASED FILTERING
        </div>
    </div>

    """, unsafe_allow_html=True)

with h_col2:
    if not st.session_state["is_admin_logged_in"]:
        if st.button("Đăng nhập", key="top_admin_btn"):
            st.session_state["show_top_admin_login"] = not st.session_state.get("show_top_admin_login", False)
            st.rerun()
    else:
        if st.button("Đăng xuất", key="top_logout_btn"):
            st.session_state["is_admin_logged_in"] = False
            st.session_state["active_view"] = "SURVEY"
            st.rerun()

# Modal / Card Đăng nhập Admin xuất hiện ngay dưới Header khi bấm nút
if not st.session_state["is_admin_logged_in"] and st.session_state.get("show_top_admin_login", False):
    with st.container(border=True):
        st.markdown("<div class='section-title' style='font-size:1.05rem !important; margin-bottom:12px !important;'>Đăng nhập Quản trị viên</div>", unsafe_allow_html=True)
        with st.form("admin_login_form"):
            f_col1, f_col2, f_col3 = st.columns([3, 1, 1])
            with f_col1:
                admin_pwd = st.text_input(
                    "Mật khẩu quản trị",
                    type="password",
                    placeholder="Nhập mật khẩu...",
                    key="admin_pwd_login",
                    autocomplete="current-password",
                    help="Nhập mật khẩu rồi bấm Enter để vào nhanh"
                )
            with f_col2:
                st.markdown("<div style='height:28px;'></div>", unsafe_allow_html=True)
                login_btn = st.form_submit_button("Vào Admin", type="primary", use_container_width=True)
            with f_col3:
                st.markdown("<div style='height:28px;'></div>", unsafe_allow_html=True)
                cancel_btn = st.form_submit_button("Đóng", use_container_width=True)

            if cancel_btn:
                st.session_state["show_top_admin_login"] = False
                st.rerun()

            if login_btn:
                pwd_clean = admin_pwd.strip()
                if pwd_clean in ["admin", "admin123", "123456"]:
                    st.session_state["is_admin_logged_in"] = True
                    st.session_state["show_top_admin_login"] = False
                    st.session_state["active_view"] = "ADMIN"
                    st.rerun()
                elif not pwd_clean:
                    st.warning("Vui lòng nhập mật khẩu.")
                else:
                    st.error("Mật khẩu không chính xác.")


# Thanh chuyển View nếu Admin đã đăng nhập hoặc đã có kết quả
if st.session_state["is_admin_logged_in"]:
    st.markdown("<div style='margin-bottom:16px;'>", unsafe_allow_html=True)
    v1, v2, v3 = st.columns(3)
    with v1:
        if st.button("Khảo sát nhu cầu", use_container_width=True):
            st.session_state["active_view"] = "SURVEY"
            st.rerun()
    with v2:
        if st.button("Xem gợi ý theo 3 Miền", use_container_width=True):
            st.session_state["active_view"] = "RECOMMENDATIONS"
            st.rerun()
    with v3:
        if st.button("Quản trị Admin", type="primary", use_container_width=True):
            st.session_state["active_view"] = "ADMIN"
            st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)
elif st.session_state.get("has_run", False):
    st.markdown("<div style='margin-bottom:16px;'>", unsafe_allow_html=True)
    v1, v2 = st.columns(2)
    with v1:
        if st.button("Sửa khảo sát nhu cầu", use_container_width=True):
            st.session_state["active_view"] = "SURVEY"
            st.rerun()
    with v2:
        if st.button("Xem gợi ý theo 3 Miền", type="primary", use_container_width=True):
            st.session_state["active_view"] = "RECOMMENDATIONS"
            st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)


# =========================================================
# GIAO DIỆN 1: KHẢO SÁT NHU CẦU (COMPACT FORM)
# =========================================================
if st.session_state["active_view"] == "SURVEY":
    st.markdown("""
    <div class="hero">
        <span class="hero-eyebrow">Tối ưu hóa Lộ trình</span>
        <div class="hero-title">THIẾT KẾ CHUYẾN ĐI CỦA RIÊNG BẠN</div>
        <div class="hero-description">
            Chia sẻ nguyện vọng và ngân sách để hệ thống gợi ý những điểm đến phù hợp nhất dành cho bạn.
        </div>
    </div>
    """, unsafe_allow_html=True)

    with st.container(border=True):
        st.markdown('<div class="section-title">Khảo sát thông tin & ràng buộc</div>', unsafe_allow_html=True)

        # 1. Ô nhập từ khóa / câu văn tự do
        style_input = st.text_input(
            "Phong cách & yêu cầu du lịch (nhập từ khóa hoặc câu văn tự do)",
            value="",
            placeholder="Ví dụ: Tôi muốn đi đến địa điểm có phong cảnh thiên nhiên xanh mát...",
            help="Nhập sở thích hoặc yêu cầu cụ thể của bạn về chuyến đi."
        )

        parsed_kws, boosts = extract_intent_keywords(style_input)
        if parsed_kws:
            st.caption(f"Đã nhận diện các từ khóa: {', '.join(parsed_kws)} - Hệ thống sẽ tự động ưu tiên.")

        st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)

        # 2. Hàng thông tin ràng buộc chuyến đi (Ngân sách & Số ngày & Số người)
        rc1, rc2, rc3 = st.columns(3)
        with rc1:
            num_people_input = st.number_input(
                "Số người đi",
                min_value=1,
                max_value=50,
                value=1,
                step=1,
                help="Nhập số lượng người tham gia chuyến đi."
            )
        with rc2:
            budget_input = st.number_input(
                "Ngân sách / người (VNĐ)",
                min_value=200000,
                max_value=50000000,
                value=3000000,
                step=500000,
                help="Nhập chi phí dự kiến cho một người. Hệ thống sẽ tự động nhân với số người."
            )
        with rc3:
            duration = st.number_input(
                "Thời gian chuyến đi (ngày)",
                min_value=0.5,
                max_value=14.0,
                value=2.0,
                step=0.5,
                help="Chọn thời gian: 0.5 ngày (nửa ngày), 1 ngày (đi về trong ngày - không ở khách sạn), hoặc từ 1.5 ngày trở lên nếu có qua đêm."
            )
            if duration <= 1.0:
                st.caption("Chuyến đi trong ngày (không cần lưu trú khách sạn).")
            else:
                _n_prev = max(0, int(np.ceil(duration)) - 1)
                st.caption(f"Chuyến đi qua đêm (dự kiến lưu trú {_n_prev} đêm).")



        st.markdown("<hr style='margin: 20px 0; border-color: var(--border);'>", unsafe_allow_html=True)

        if st.button("Phân tích & Tìm kiếm gợi ý", type="primary", use_container_width=True):
            # Trọng số khởi điểm theo có/không có mô tả sở thích. Khi đủ phản hồi người dùng,
            # trang gợi ý sẽ học lại các trọng số này (utils/learning.py).
            rec_weights = recommended_weights(bool(style_input.strip()))

            st.session_state["user_inputs"] = {
                "num_people": num_people_input,
                "budget": budget_input,
                "duration": duration,
                "style": style_input,
                "parsed_kws": parsed_kws
            }
            st.session_state["topsis_weights"] = rec_weights
            st.session_state["has_run"] = True
            st.session_state["active_view"] = "RECOMMENDATIONS"
            st.rerun()


# =========================================================
# GIAO DIỆN 2: GỢI Ý THEO 3 MIỀN & ĐÁNH GIÁ (IMAGE 1 STYLE)
# =========================================================
elif st.session_state["active_view"] == "RECOMMENDATIONS":
    if not st.session_state.get("has_run", False):
        st.warning("Bạn chưa nhập thông tin khảo sát. Vui lòng hoàn thành khảo sát nhu cầu trước.")
        if st.button("Chuyển tới khảo sát nhu cầu"):
            st.session_state["active_view"] = "SURVEY"
            st.rerun()
    else:
        u_inputs = st.session_state.get("user_inputs", {})
        num_people = u_inputs.get("num_people", 1)
        budget_per_person = u_inputs.get("budget", 3000000)
        total_budget = budget_per_person * num_people
        duration = u_inputs.get("duration", 2.0)
        style = u_inputs.get("style", "")
        weights = st.session_state.get("topsis_weights") or recommended_weights(bool(style.strip()))

        # Hero Banner
        st.markdown("""
        <div class="hero">
            <div class="hero-content">
                <div class="hero-title">GỢI Ý DANH SÁCH ĐIỂM ĐẾN DU LỊCH THEO 3 MIỀN</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Bước 2-3: học từ hành vi người dùng trước (phản hồi tường minh)
        _reviews = load_feedback_data().get("explicit_reviews", [])
        community, fb_counts = community_scores(_reviews, build_name_index(dest_df))
        weights, learn_info = learn_weights(_reviews, weights, has_query=bool(style.strip()))

        scored_df = rank_by_suitability(
            result_df,
            weights=weights,
            budget=total_budget,
            num_people=num_people,
            duration=duration,
            preferred_duration=(max(0.5, duration - 1), duration + 1),
            keyword_query=style,
            tfidf_model=vectorizer,
            tfidf_matrix=tfidf_matrix,
            community=community,
            hotel_df=hotel_df,
            rest_df=rest_df
        )

        # Region Tabs
        st.markdown('<div class="section-title">Gợi ý điểm đến hàng đầu theo 3 Miền</div>', unsafe_allow_html=True)
        nights_calc = max(0, int(np.ceil(duration)) - 1) if duration > 1.0 else 0
        meals_calc = 1 if duration <= 0.5 else (2 if duration <= 1.0 else max(2, int(round(duration * 2))))
        stay_desc = f"khách sạn ({num_people} người · {nights_calc} đêm)" if nights_calc > 0 else "không qua đêm (chuyến đi trong ngày)"
        st.info(f"Dự toán ngân sách chuyến đi: Đã tổng hợp vé tham quan, {stay_desc} và ẩm thực nhà hàng (~{meals_calc} bữa) để đối chiếu với ngân sách {total_budget/1e6:.1f} triệu VNĐ của nhóm bạn.")
        t_mb, t_mt, t_mn = st.tabs(["Miền Bắc", "Miền Trung", "Miền Nam"])

        def render_mini_card(title, rating_num, price_min, price_max, price_suffix):
            rate_html = ""
            if pd.notna(rating_num):
                try:
                    n = float(rating_num)
                    if not np.isnan(n) and n > 0:
                        rate_html = f"<span style='color:#F5A623; font-weight:bold;'>★</span> <span style='font-weight:600;'>{n:.1f}</span>"
                except (ValueError, TypeError):
                    pass

            p_str = format_price_range(price_min, price_max)
            p_html = f"<span style='color:var(--primary); font-weight:600;'>{p_str}{price_suffix}</span>" if p_str else ""

            meta_elements = [el for el in [rate_html, p_html] if el]
            meta_row = " <span style='color:#DDD5CE; margin:0 4px;'>•</span> ".join(meta_elements) if meta_elements else ""

            return f"""
<div style="background:#FAF8F5; border:1px solid #ECE3D9; border-radius:6px; padding:7px 11px; margin-bottom:7px;">
    <div style="font-weight:600; font-size:0.86rem; color:var(--text-main); line-height:1.35; margin-bottom:2px;">{title}</div>
    <div style="font-size:0.82rem; color:var(--text-2); display:flex; align-items:center; flex-wrap:wrap; gap:4px;">{meta_row}</div>
</div>
"""

        def render_region_destinations(region_name, df_data):
            region_df = df_data[df_data["Miền"] == region_name].copy()
            if region_df.empty:
                st.info(f"Không tìm thấy địa điểm nào ở {region_name}.")
                return

            top_region = region_df.head(6)
            for idx, row in top_region.iterrows():
                score = row.get("suitability_score", 85.0)
                banner_html = safe_image_tag(row.get("image_url"))
                cluster_clean = clean_title_icon(row.get("cluster_name", "Cụm địa điểm"))

                r_num = row.get("rating_num")
                rating_part = f"<span><span style='color:#F5A623; font-size:0.92rem;'>★</span> <b style='color:var(--text-main);'>{r_num:.1f}</b>/5</span><span class='meta-sep'>•</span>" if (pd.notna(r_num) and r_num > 0) else ""
                province_part = f"<span>{row.get('province', '')}</span>" if row.get('province') else ""
                price_str = format_price_range(row.get('cost_min'), row.get('cost_max'))
                tot_est = row.get("est_total_cost", 0.0)
                tot_per_person = tot_est / num_people if num_people > 0 else tot_est
                if tot_per_person >= 1e6:
                    cost_fmt = f"~{tot_per_person/1e6:.2f} tr/người"
                else:
                    cost_fmt = f"~{tot_per_person:,.0f} đ/người"
                cost_part = f"<span class='meta-sep'>•</span><span style='color:var(--primary); font-weight:600;' title='Dự toán toàn chuyến gồm Vé + Khách sạn + Ẩm thực'>Dự toán: {cost_fmt}</span>" if tot_est > 0 else ""

                # Mô tả địa điểm có (Xem thêm / Thu gọn)
                desc_text = str(row.get('description', '')).strip()
                if len(desc_text) > 130:
                    desc_intro = desc_text[:125].rsplit(' ', 1)[0]
                    desc_html = f"""<div style="font-size:0.88rem; color:var(--text-2); line-height:1.55; margin-bottom:8px;">
<details class="desc-details">
<summary>
<span class="short-text">{desc_intro}... <span style="color:var(--primary); font-weight:600; text-decoration:underline; font-size:0.84rem;">(Xem thêm)</span></span>
<span class="show-less">{desc_text} <span style="color:var(--primary); font-weight:600; text-decoration:underline; font-size:0.84rem; margin-left:4px;">(Thu gọn)</span></span>
</summary>
</details>
</div>"""
                else:
                    desc_html = f"""<div style="font-size:0.88rem; color:var(--text-2); line-height:1.55; margin-bottom:8px;">{desc_text}</div>"""

                col_img, col_info = st.columns([1, 2])
                with col_img:
                    st.markdown(banner_html, unsafe_allow_html=True)
                with col_info:
                    card_html = f"""<div class="dest-name">{row['destination_name'].strip()}</div>
<div class="dest-meta">
<span class="match-badge">{score:.0f}% Phù hợp</span>
{rating_part}
{province_part}
{cost_part}
</div>
{desc_html}
<span class="cluster-tag">{cluster_clean}</span>"""
                    st.markdown(card_html, unsafe_allow_html=True)

                    with st.expander(f"Chi tiết chi phí & gợi ý cho {row['destination_name'].strip()}"):
                        # Log implicit view
                        _logged = st.session_state.setdefault("_logged_views", set())
                        if row["destination_id"] not in _logged:
                            log_implicit_view(row["destination_id"], row["destination_name"].strip(), region_name)
                            _logged.add(row["destination_id"])

                        reasons = explain_top_criteria(
                            row, budget=total_budget, keyword_query=style,
                            feedback=fb_counts.get(row["destination_id"]),
                        )
                        if reasons:
                            items_li = []
                            for r in reasons:
                                r_clean = r.replace("★", "<span style='color:#F5A623; font-weight:bold;'>★</span>")
                                items_li.append(f"<li style='margin-bottom:4px;'>{r_clean}</li>")
                            st.markdown(f"""
<div style="margin-bottom:12px;">
    <div style="font-size:0.88rem; font-weight:600; color:var(--primary); margin-bottom:5px;">Vì sao lựa chọn này phù hợp?</div>
    <ul style="margin:0; padding-left:18px; font-size:0.85rem; color:var(--text-2); line-height:1.5;">
        {''.join(items_li)}
    </ul>
</div>
""", unsafe_allow_html=True)

                        ticket_val = row.get('est_ticket_cost', 0)
                        hotel_val = row.get('est_hotel_cost', 0)
                        food_val = row.get('est_food_cost', 0)
                        tot_val = row.get('est_total_cost', 0)

                        hotel_text = f"Khách sạn ({nights_calc} đêm): <b style='color:var(--text-main);'>{hotel_val:,.0f} đ</b>" if nights_calc > 0 else "Lưu trú: <b style='color:var(--text-main);'>0 đ (Đi trong ngày)</b>"
                        dur_lbl = f"{duration:g} ngày (nửa ngày)" if duration == 0.5 else f"{duration:g} ngày"

                        st.markdown(f"""
<div style="background:#FAF8F5; border:1px solid #EAE1D7; border-radius:6px; padding:10px 14px; margin:10px 0 14px 0;">
    <div style="display:flex; justify-content:space-between; align-items:center; border-bottom:1px solid #EFE7DE; padding-bottom:6px; margin-bottom:8px;">
        <span style="font-weight:600; font-size:0.88rem; color:var(--text-main);">Dự toán chi phí ({num_people} người · {dur_lbl})</span>
        <span style="font-weight:700; color:var(--primary); font-size:1.0rem;">~{tot_val:,.0f} VNĐ</span>
    </div>
    <div style="display:flex; flex-wrap:wrap; gap:12px; font-size:0.84rem; color:var(--text-2); align-items:center;">
        <span>Vé tham quan: <b style="color:var(--text-main);">{ticket_val:,.0f} đ</b></span>
        <span style="color:#D6CBC2;">•</span>
        <span>Ẩm thực (~{meals_calc} bữa): <b style="color:var(--text-main);">{food_val:,.0f} đ</b></span>
        <span style="color:#D6CBC2;">•</span>
        <span>{hotel_text}</span>
    </div>
</div>
""", unsafe_allow_html=True)

                        if nights_calc > 0:
                            c_hotel, c_rest = st.columns(2)
                            with c_hotel:
                                st.markdown("<div style='font-size:0.86rem; font-weight:600; color:var(--text-main); margin-bottom:6px;'>Khách sạn lưu trú qua đêm</div>", unsafe_allow_html=True)
                                hotels = hotel_df[hotel_df["destination_id"] == row["destination_id"]].head(2)
                                if hotels.empty:
                                    st.caption("Đang cập nhật khách sạn...")
                                for _, h in hotels.iterrows():
                                    st.markdown(render_mini_card(
                                        clean_title_icon(h.get('hotel_name', '')),
                                        h.get('rating_num'),
                                        h.get('price_min'),
                                        h.get('price_max'),
                                        "/đêm"
                                    ), unsafe_allow_html=True)
                            with c_rest:
                                st.markdown("<div style='font-size:0.86rem; font-weight:600; color:var(--text-main); margin-bottom:6px;'>Quán ăn gợi ý</div>", unsafe_allow_html=True)
                                rests = rest_df[rest_df["destination_id"] == row["destination_id"]].head(2)
                                if rests.empty:
                                    st.caption("Đang cập nhật quán ăn...")
                                for _, r in rests.iterrows():
                                    st.markdown(render_mini_card(
                                        clean_title_icon(r.get('restaurant_name', '')),
                                        r.get('rating_num'),
                                        r.get('price_min'),
                                        r.get('price_max'),
                                        "/người"
                                    ), unsafe_allow_html=True)
                        else:
                            st.markdown("<div style='font-size:0.86rem; font-weight:600; color:var(--text-main); margin-bottom:6px;'>Quán ăn & Ẩm thực lân cận (Chuyến đi trong ngày, không cần lưu trú)</div>", unsafe_allow_html=True)
                            rests = rest_df[rest_df["destination_id"] == row["destination_id"]].head(3)
                            if rests.empty:
                                st.caption("Đang cập nhật quán ăn...")
                            for _, r in rests.iterrows():
                                st.markdown(render_mini_card(
                                    clean_title_icon(r.get('restaurant_name', '')),
                                    r.get('rating_num'),
                                    r.get('price_min'),
                                    r.get('price_max'),
                                    "/người"
                                ), unsafe_allow_html=True)


                st.markdown("<hr style='margin: 16px 0; border-color: var(--border);'>", unsafe_allow_html=True)

        with t_mb:
            render_region_destinations("Miền Bắc", scored_df)
        with t_mt:
            render_region_destinations("Miền Trung", scored_df)
        with t_mn:
            render_region_destinations("Miền Nam", scored_df)

        # =========================================================
        # KHỐI ĐÁNH GIÁ NGƯỜI DÙNG (IMAGE 1 STYLE)
        # =========================================================
        with st.container(border=True):
            st.markdown('<div class="section-title">Kết quả gợi ý có hữu ích không?</div>', unsafe_allow_html=True)
            st.markdown('<div style="font-size:0.88rem; color:var(--text-muted); margin-bottom:14px;">Đánh giá của bạn giúp hệ thống cập nhật trọng số và cải thiện độ chính xác.</div>', unsafe_allow_html=True)

            st.markdown("<div style='font-size:0.86rem; font-weight:600; color:var(--text-main); margin-bottom:4px;'>Mức độ hài lòng:</div>", unsafe_allow_html=True)
            stars_choice = st.radio(
                "Mức độ hài lòng",
                [1, 2, 3, 4, 5],
                index=None,
                format_func=lambda x: f"{x} sao",
                horizontal=True,
                label_visibility="collapsed"
            )

            st.markdown("<div style='font-size:0.86rem; font-weight:600; color:var(--text-main); margin-top:14px; margin-bottom:6px;'>Đánh giá sự phù hợp của các địa điểm gợi ý:</div>", unsafe_allow_html=True)

            top_recommended = scored_df.head(5)
            item_evals = {}
            for _, r_item in top_recommended.iterrows():
                d_name = r_item["destination_name"].strip()
                d_region = r_item["Miền"]
                display_title = f"{d_name} ({d_region})"

                ic1, ic2 = st.columns([2, 1])
                with ic1:
                    st.markdown(f"• <b>{display_title}</b>", unsafe_allow_html=True)
                with ic2:
                    choice = st.radio(
                        f"Fit_{r_item['destination_id']}",
                        ["Phù hợp", "Chưa phù hợp"],
                        index=None,
                        horizontal=True,
                        label_visibility="collapsed"
                    )
                    item_evals[display_title] = choice

            comment_input = st.text_area(
                "Góp ý thêm (không bắt buộc)",
                placeholder="Nhập ý kiến nhận xét của bạn về các địa điểm gợi ý..."
            )

            if st.button("Gửi đánh giá", type="primary"):
                # Không đặt đáp án mặc định: nhãn "Phù hợp" mặc định sẽ làm Precision@5 cao giả tạo
                if stars_choice is None or any(v is None for v in item_evals.values()):
                    st.warning("Vui lòng chọn mức hài lòng và đánh giá đủ từng địa điểm (Phù hợp / Chưa phù hợp) trước khi gửi.")
                else:
                    fit_list = [k for k, v in item_evals.items() if v == "Phù hợp"]
                    unfit_list = [k for k, v in item_evals.items() if v == "Chưa phù hợp"]
                    persona_label = f"Khách (Ngân sách: {budget_per_person/1e6:.1f} triệu)"

                    # Ghi kèm đặc trưng TOPSIS của từng địa điểm để hệ thống học lại trọng số
                    _feat_cols = ["rating_num", "budget_fit", "duration_fit", "style_match",
                                  "cluster_fit", "community_score"]
                    items_payload = []
                    for rank_i, (_, r_item) in enumerate(top_recommended.iterrows(), start=1):
                        _title = f"{r_item['destination_name'].strip()} ({r_item['Miền']})"
                        items_payload.append({
                            "dest_id": r_item["destination_id"],
                            "name": r_item["destination_name"].strip(),
                            "rank": rank_i,
                            "label": "fit" if item_evals.get(_title) == "Phù hợp" else "unfit",
                            "features": {c: float(r_item[f"f_{c}"]) for c in _feat_cols},
                        })

                    save_explicit_feedback(
                        persona=persona_label,
                        stars=stars_choice,
                        fit_items=fit_list,
                        unfit_items=unfit_list,
                        comment=comment_input,
                        items=items_payload,
                        context={"budget": budget_per_person, "duration": duration, "style": style},
                    )
                    st.success("Cảm ơn bạn! Đánh giá đã được lưu thành công vào hệ thống.")


# =========================================================
# GIAO DIỆN 3: QUẢN TRỊ ADMIN - PHẢN HỒI NGƯỜI DÙNG (IMAGE 2 STYLE)
# =========================================================
elif st.session_state["active_view"] == "ADMIN":
    if not st.session_state.get("is_admin_logged_in", False):
        st.error("Bạn chưa đăng nhập quyền Quản trị viên. Vui lòng bấm nút 'Đăng nhập' ở góc trên bên phải.")
        st.stop()

    st.markdown("""
    <div class="hero">
        <div class="hero-content">
            <span class="hero-eyebrow">Khu vực Quản trị</span>
            <div class="hero-title">PHẢN HỒI NGƯỜI DÙNG & TƯƠNG TÁC THỰC TẾ</div>
            <div class="hero-description">
                Tổng hợp Explicit Feedback (đánh giá tường minh) và Implicit Feedback (hành vi mở xem) thu thập từ hệ thống.
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    stats = get_feedback_stats()

    if is_cloud_configured():
        st.success("**Lưu trữ Cloud (Google Sheets):** Đã kết nối thành công. Toàn bộ đánh giá từ người dùng được lưu vĩnh viễn và không bị mất khi ứng dụng khởi động lại.")
    else:
        st.warning("**Lưu trữ Cục bộ (Chưa kết nối Google Sheets):** Dữ liệu đang lưu tạm trên file server. Khi app chuyển sang chế độ Sleep trên Streamlit Cloud, dữ liệu có thể bị xóa. Xem hướng dẫn kết nối Google Sheets ở mục Quản lý phía dưới.")

    # Metric KPI Cards
    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-value">{stats['total_explicit']}</div>
            <div class="kpi-label">LƯỢT ĐÁNH GIÁ TƯỜNG MINH</div>
        </div>
        """, unsafe_allow_html=True)
    with m2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-value">{stats['avg_stars']} / 5</div>
            <div class="kpi-label">ĐIỂM TRUNG BÌNH</div>
        </div>
        """, unsafe_allow_html=True)
    with m3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-value">{stats['total_views']}</div>
            <div class="kpi-label">LƯỢT MỞ XEM CHI TIẾT</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height:24px;'></div>", unsafe_allow_html=True)

    # 0. Đánh giá hệ thống (bước 7 của quy trình)
    with st.container(border=True):
        st.markdown('<div class="section-title">Đánh giá hệ thống</div>', unsafe_allow_html=True)

        om = online_metrics(stats["explicit_reviews"])
        clu = clustering_report(feature_matrix, result_df["cluster_id"].values)
        _rv = stats["explicit_reviews"]
        _comm, _ = community_scores(_rv, build_name_index(dest_df))
        _, _linfo = learn_weights(_rv, recommended_weights(True), has_query=True)

        def _pct(v):
            return "—" if v is None else f"{v * 100:.0f}%"

        _learn_txt = (f"ĐÃ HỌC ({_linfo['n_labels']} nhãn)" if _linfo["status"] == "learned"
                      else f"CHỜ DỮ LIỆU ({_linfo['n_labels']}/{MIN_LABELS} nhãn)")
        ev_cards = [
            (_pct(om["precision_at_5"]), "PRECISION@5 (NGƯỜI DÙNG THẬT)"),
            (_pct(om["hit_rate"]), "HIT-RATE@5 (≥1 ĐỊA ĐIỂM PHÙ HỢP)"),
            (f"{clu['silhouette']:.2f} · k={clu['k']}", "SILHOUETTE K-MEANS · SỐ CỤM"),
            (_learn_txt, "TRỌNG SỐ HỌC TỪ PHẢN HỒI"),
        ]
        for col, (val, lab) in zip(st.columns(4), ev_cards):
            with col:
                st.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-value" style="font-size:1.35rem;">{val}</div>
                    <div class="kpi-label">{lab}</div>
                </div>
                """, unsafe_allow_html=True)

        st.caption(
            f"Cụm K-Means có {min(clu['sizes'])}–{max(clu['sizes'])} điểm đến "
            f"(kích thước: {', '.join(map(str, clu['sizes']))}). "
            + (f"AUC kiểm định chéo của mô hình học trọng số: {_linfo['cv_auc']:.2f}. "
               if _linfo.get("cv_auc") else "")
            + "Precision@5 = tỉ lệ địa điểm được người dùng đánh giá 'Phù hợp' trong top 5."
        )

        st.markdown("<b style='font-size:0.80rem; text-transform:uppercase;'>So sánh với hệ thống cơ sở (đánh giá offline)</b>", unsafe_allow_html=True)
        eval_res = run_offline_eval(
            result_df, vectorizer, tfidf_matrix, tuple(sorted(_comm.items())),
            _hotel_df=hotel_df, _rest_df=rest_df
        )
        if isinstance(eval_res, tuple) and len(eval_res) == 3:
            eval_tbl, n_scn, eval_stats = eval_res
        elif isinstance(eval_res, tuple) and len(eval_res) == 2:
            eval_tbl, n_scn = eval_res
            eval_stats = {}
        else:
            eval_tbl, n_scn, eval_stats = eval_res[0], eval_res[1], {}

        st.dataframe(eval_tbl, use_container_width=True, hide_index=True)
        sig_info = ""
        if eval_stats and "p_value" in eval_stats and eval_stats["p_value"] is not None:
            sig_info = f" (Kiểm định Wilcoxon so với baseline '{eval_stats.get('baseline')}': p = {eval_stats['p_value']:.4f})."
        st.caption(
            f"Mô phỏng {n_scn} kịch bản (ngân sách × số ngày × chủ đề). Một địa điểm 'phù hợp' khi thỏa các ràng buộc "
            "khách nêu: chi phí tối đa ≤ ngân sách, thời lượng lệch ≤ 1 ngày, và đúng chủ đề theo tên/từ khóa gán tay "
            f"(độc lập với TF-IDF của bộ xếp hạng){sig_info}. Dòng 'Giới hạn trên' là mức tối đa có thể đạt. "
            "Đây là đánh giá theo yêu cầu đã nêu, không thay thế đánh giá của người dùng thật."
        )

    st.markdown("<div style='height:24px;'></div>", unsafe_allow_html=True)

    # Trực quan hóa Dữ liệu Thực nghiệm (Sơ đồ & Biểu đồ)
    with st.container(border=True):
        st.markdown('<div class="section-title">Trực quan hóa Dữ liệu Thực nghiệm (Sơ đồ & Biểu đồ)</div>', unsafe_allow_html=True)
        if stats["total_explicit"] > 0:
            c_chart1, c_chart2 = st.columns(2)
            with c_chart1:
                st.markdown("<div style='font-size:0.88rem; font-weight:600; color:var(--text-main); margin-bottom:8px;'>Phân bố mức độ hài lòng (1 - 5 sao)</div>", unsafe_allow_html=True)
                star_counts = {f"{i} sao": 0 for i in range(1, 6)}
                for r in stats["explicit_reviews"]:
                    s = r.get("stars")
                    if s in [1, 2, 3, 4, 5]:
                        star_counts[f"{s} sao"] += 1
                df_stars = pd.DataFrame(list(star_counts.items()), columns=["Mức độ", "Số lượt"])
                st.bar_chart(df_stars.set_index("Mức độ"), color="#8C2D19")

            with c_chart2:
                st.markdown("<div style='font-size:0.88rem; font-weight:600; color:var(--text-main); margin-bottom:8px;'>Sự dịch chuyển Trọng số TOPSIS (Ban đầu vs Đã học)</div>", unsafe_allow_html=True)
                FEATURE_NAMES = {
                    "rating_num": "Đánh giá (Rating)",
                    "budget_fit": "Ngân sách (Budget)",
                    "duration_fit": "Thời gian (Duration)",
                    "style_match": "Sở thích (Style)",
                    "cluster_fit": "Cụm K-Means"
                }
                base_w = recommended_weights(True)
                learned_w, _ = learn_weights(stats["explicit_reviews"], base_w, has_query=True)
                df_weights = pd.DataFrame({
                    "Tiêu chí": [FEATURE_NAMES[k] for k in LEARN_FEATURES],
                    "Trọng số ban đầu": [base_w.get(k, 0.0) for k in LEARN_FEATURES],
                    "Trọng số đã học": [learned_w.get(k, 0.0) for k in LEARN_FEATURES]
                })
                st.bar_chart(df_weights.set_index("Tiêu chí"), color=["#B3ABA5", "#8C2D19"])

            # Top địa điểm phù hợp nhất
            fit_counter = {}
            for r in stats["explicit_reviews"]:
                for item in r.get("fit_items", []):
                    fit_counter[item] = fit_counter.get(item, 0) + 1
            if fit_counter:
                st.markdown("<div style='font-size:0.88rem; font-weight:600; color:var(--text-main); margin-top:16px; margin-bottom:8px;'>Top địa điểm được người dùng đánh giá 'Phù hợp' nhiều nhất</div>", unsafe_allow_html=True)
                top_fits = sorted(fit_counter.items(), key=lambda x: x[1], reverse=True)[:8]
                df_top_fit = pd.DataFrame(top_fits, columns=["Điểm đến", "Lượt vote"]).set_index("Điểm đến")
                st.bar_chart(df_top_fit, color="#8C2D19")
        else:
            st.info("Chưa có đủ dữ liệu đánh giá để vẽ biểu đồ. Khi người dùng gửi đánh giá từ trang Gợi ý, hệ thống sẽ tự động vẽ sơ đồ trực quan hóa tại đây.")

    st.markdown("<div style='height:24px;'></div>", unsafe_allow_html=True)

    # 1. Implicit Feedback Table
    with st.container(border=True):
        st.markdown('<div class="section-title">Lượt xem địa điểm (Implicit Feedback)</div>', unsafe_allow_html=True)

        implicit_data = stats["implicit_list"]
        if implicit_data:
            df_imp = pd.DataFrame(implicit_data)
            df_imp.columns = ["Mã địa điểm", "Tên địa điểm", "Miền", "Số lượt mở xem"]
            st.dataframe(df_imp, use_container_width=True, hide_index=True)
        else:
            st.info("Chưa có ghi nhận tương tác ngầm nào.")

    # 2. Explicit Feedback Table
    with st.container(border=True):
        st.markdown('<div class="section-title">Đánh giá chi tiết (Explicit Feedback)</div>', unsafe_allow_html=True)

        explicit_data = stats["explicit_reviews"]
        if explicit_data:
            formatted_rows = []
            for r in explicit_data:
                formatted_rows.append({
                    "Thời gian": r["timestamp"],
                    "Ngữ cảnh khảo sát": r["persona"],
                    "Sao": f"{r['stars']} / 5 sao",
                    "Phù hợp": ", ".join(r["fit_items"]) if r["fit_items"] else "—",
                    "Chưa phù hợp": ", ".join(r["unfit_items"]) if r["unfit_items"] else "—",
                    "Bình luận": r["comment"] if r["comment"] else "—"
                })
            df_exp = pd.DataFrame(formatted_rows)
            st.dataframe(df_exp, use_container_width=True, hide_index=True)
        else:
            st.info("Chưa có lượt đánh giá tường minh nào.")

    st.markdown("<div style='height:24px;'></div>", unsafe_allow_html=True)

    # 3. Quản lý dữ liệu thực nghiệm & Đồng bộ Cloud
    with st.container(border=True):
        st.markdown('<div class="section-title">Quản lý dữ liệu thực nghiệm & Đồng bộ Cloud</div>', unsafe_allow_html=True)
        st.caption("Sao lưu file JSON, đồng bộ dữ liệu lên Google Sheets hoặc xóa sạch dữ liệu để thực nghiệm lại từ đầu.")

        q_col1, q_col2, q_col3 = st.columns(3)
        with q_col1:
            feedback_json_str = json.dumps(load_feedback_data(), ensure_ascii=False, indent=2)
            st.download_button(
                label="Tải file sao lưu (.json)",
                data=feedback_json_str,
                file_name="feedback_backup.json",
                mime="application/json",
                use_container_width=True
            )
        with q_col2:
            if is_cloud_configured():
                if st.button("Đồng bộ lên Google Sheets", use_container_width=True):
                    ok, msg = sync_local_to_cloud()
                    if ok:
                        st.success(msg)
                        st.rerun()
                    else:
                        st.error(msg)
            else:
                st.button("Chưa kết nối Google Sheets", disabled=True, use_container_width=True)
        with q_col3:
            if st.button("Xóa sạch dữ liệu (Reset)", type="secondary", use_container_width=True):
                reset_feedback_data()
                st.session_state.pop("_logged_views", None)
                st.success("Đã xóa toàn bộ dữ liệu phản hồi về trạng thái ban đầu!")
                st.rerun()

        with st.expander("Hướng dẫn kết nối Google Sheets để lưu đánh giá vĩnh viễn (Chỉ mất 1 phút)"):
            st.markdown("""
            **Cách thiết lập để đánh giá của mọi người không bao giờ bị mất khi deploy Streamlit Cloud:**

            1. Mở [Google Sheets](https://sheets.new) tạo một file Trang tính mới.
            2. Trên thanh menu, chọn **Tiện ích mở rộng (Extensions)** -> **Apps Script**.
            3. Xóa hết mã có sẵn trong Apps Script, dán toàn bộ mã trong file [`google_sheets_script.js`](file:///c:/Users/Lenovo/Downloads/DSS%20%281%29/DSS/google_sheets_script.js) vào.
            4. Bấm **Triển khai (Deploy)** -> **Tùy chọn triển khai mới (New deployment)**.
            5. Bấm vào biểu tượng bánh răng bên cạnh 'Chọn loại' -> Chọn **Ứng dụng web (Web app)**:
               - **Mô tả:** `DSS Feedback Backend`
               - **Thực thi dưới dạng:** `Tôi (tài khoản của bạn)`
               - **Người có quyền truy cập:** **Bất kỳ ai (Anyone)** *(Quan trọng: phải chọn Bất kỳ ai để Streamlit gửi được dữ liệu)*
            6. Bấm **Triển khai (Deploy)** -> Bấm **Ủy quyền truy cập (Authorize access)** -> Sao chép **URL ứng dụng web** (có đuôi `/exec`).
            7. **Cấu hình trên Streamlit Cloud:**
               - Vào dashboard của [Streamlit Cloud](https://share.streamlit.io) -> Bấm vào dấu `...` cạnh app -> Chọn **Settings** -> **Secrets**.
               - Dán dòng sau vào và bấm Save:
                 ```toml
                 FEEDBACK_WEBHOOK_URL = "URL_BẠN_VỪA_COPY"
                 ```
            8. **Nếu muốn test trên máy tính:** Dán dòng trên vào file `.streamlit/secrets.toml`.
            """)
