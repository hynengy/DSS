import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
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
from utils.learning import build_name_index, community_scores, learn_weights, MIN_LABELS
from utils.evaluation import offline_evaluation, online_metrics, clustering_report


from utils.feedback_manager import (
    load_feedback_data,
    log_implicit_view,
    save_explicit_feedback,
    get_feedback_stats,
)
from sklearn.metrics import silhouette_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Hệ thống Trợ giúp Ra quyết định Du lịch Việt Nam",
    page_icon="🇻🇳",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# =========================================================
# CUSTOM CSS (ENTERPRISE MONTSERRAT DESIGN SYSTEM)
# =========================================================
st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;500;600;700;800&family=Playfair+Display:ital,wght@0,400;0,600;0,700;1,400;1,600&display=swap');

:root {
    --primary:      #9E4429;
    --primary-dk:   #81351E;
    --primary-lt:   #FBF4F1;
    --bg-base:      #F7F5EF;
    --bg-card:      #FFFFFF;
    --cream-dk:     #EFEBE2;
    --text-main:    #242019;
    --text-muted:   #6C6556;
    --text-2:       #595346;
    --navy:         #242019;
    --navy-mid:     #474034;
    --white:        #FFFFFF;
    --border:       #DFD8CB;
    --border-light: #ECE6DA;
    --gold:         #B87C28;
    --gold-bg:      #FCF8EF;
    --shadow-sm:    0 2px 8px rgba(36, 32, 25, 0.04);
    --shadow-md:    0 8px 24px rgba(36, 32, 25, 0.07);
    --shadow-lg:    0 16px 36px rgba(36, 32, 25, 0.10);
    --r-sm:         8px;
    --r-md:         14px;
    --font:         'Montserrat', -apple-system, BlinkMacSystemFont, sans-serif;
    --font-serif:   'Playfair Display', Georgia, serif;
}

/* Base font & page background */
html, body, .stApp {
    font-family: var(--font);
    background-color: var(--bg-base) !important;
    color: var(--text-main);
}

p, label, li, span {
    font-family: var(--font);
    color: var(--text-main);
    line-height: 1.5;
}

/* Hide Streamlit default chrome */
#MainMenu, footer, [data-testid="stDecoration"] { visibility: hidden; }

/* ═══ HEADINGS & TITLES (With safe line-height for Vietnamese accents) ═══ */
h1, h2, h3, h4, .admin-table-title {
    font-family: var(--font) !important;
    text-transform: uppercase !important;
    font-weight: 700 !important;
    color: var(--text-main) !important;
    letter-spacing: 0.05em;
    line-height: 1.4 !important;
    overflow: visible !important;
}

/* ═══ HERO BANNER (Luxury Editorial Style) ═══ */
.hero {
    background-color: #242019;
    background-image: radial-gradient(circle at top right, rgba(158, 68, 41, 0.22), transparent 55%),
                      radial-gradient(circle at bottom left, rgba(184, 124, 40, 0.15), transparent 55%);
    border-radius: var(--r-md);
    padding: 50px 36px;
    text-align: center;
    margin-bottom: 28px;
    box-shadow: var(--shadow-md);
    border: 1px solid rgba(223, 216, 203, 0.15);
}
.hero-eyebrow {
    color: var(--gold);
    font-size: 0.75rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.2em;
    margin-bottom: 12px;
    display: inline-block;
    line-height: 1.4;
}
.hero-title {
    font-family: var(--font-serif) !important;
    font-style: italic;
    font-weight: 500 !important;
    font-size: 2.7rem !important;
    line-height: 1.45 !important;
    color: #FAF7F2 !important;
    margin-bottom: 14px !important;
    overflow: visible !important;
}
.hero-description {
    color: #D6D0C4;
    font-size: 0.98rem;
    max-width: 620px;
    margin: 0 auto;
    line-height: 1.6;
}

/* ═══ CARD CONTAINERS (Clean & Un-nested) ═══ */
div[data-testid="stVerticalBlockBorderWrapper"] {
    background: var(--bg-card);
    border: 1px solid var(--border) !important;
    border-radius: var(--r-md) !important;
    box-shadow: var(--shadow-sm) !important;
    margin-bottom: 24px;
    transition: box-shadow 0.25s ease, border-color 0.25s ease;
}
div[data-testid="stVerticalBlockBorderWrapper"]:hover {
    border-color: #D0C7B6 !important;
    box-shadow: var(--shadow-md) !important;
}
div[data-testid="stVerticalBlockBorderWrapper"] > div {
    padding: 24px 28px !important;
}

.section-title {
    font-family: var(--font-serif) !important;
    font-size: 1.6rem !important;
    font-weight: 600 !important;
    color: var(--text-main) !important;
    margin-bottom: 20px !important;
    border-bottom: 1px solid var(--border);
    padding-bottom: 10px;
    line-height: 1.45 !important;
    overflow: visible !important;
}

/* ═══ BUTTONS ═══ */
.stButton > button, .stButton > button p {
    font-family: var(--font) !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
    line-height: 1.4 !important;
}
.stButton > button {
    border-radius: var(--r-sm) !important;
    padding: 8px 22px !important;
    transition: all 0.25s ease !important;
}
.stButton > button:hover {
    transform: translateY(-1px);
}

/* Align button right in header column */
div[data-testid="column"]:nth-of-type(2) .stButton {
    display: flex;
    justify-content: flex-end;
}

/* ═══ FORM LABELS & INPUTS ═══ */
.stTextInput label, .stNumberInput label, .stSlider label, .stTextArea label {
    text-transform: uppercase !important;
    font-size: 0.76rem !important;
    font-weight: 700 !important;
    color: var(--text-muted) !important;
    letter-spacing: 0.06em !important;
    line-height: 1.4 !important;
    overflow: visible !important;
}
.stSlider label p {
    white-space: normal !important;
    overflow: visible !important;
    margin-bottom: 6px !important;
}
.stSlider {
    padding-top: 4px;
    margin-bottom: 8px;
}

/* Times New Roman font inside user input */
.stTextInput input, .stTextArea textarea {
    font-family: 'Times New Roman', Times, serif !important;
    font-size: 1.12rem !important;
    color: var(--text-main) !important;
    background-color: #FFFFFF !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--r-sm) !important;
    padding: 10px 14px !important;
}
.stTextInput input::placeholder, .stTextArea textarea::placeholder {
    font-family: 'Times New Roman', Times, serif !important;
    font-style: italic !important;
    color: #9C9484 !important;
    font-size: 1.05rem !important;
}
.stTextInput input:focus, .stTextArea textarea:focus {
    border-color: var(--primary) !important;
    box-shadow: 0 0 0 2px rgba(158, 68, 41, 0.15) !important;
}

/* ═══ KPI METRIC CARDS ═══ */
.kpi-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--r-md);
    padding: 22px;
    text-align: center;
    box-shadow: var(--shadow-sm);
}
.kpi-value {
    color: var(--primary);
    font-size: 2.3rem;
    font-weight: 700;
    font-family: var(--font-serif);
    line-height: 1.3;
}
.kpi-label {
    color: var(--text-muted);
    font-size: 0.75rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-top: 6px;
    line-height: 1.4;
}

/* ═══ DESTINATION ITEM CARDS ═══ */
.dest-card-img {
    width: 100%; height: 165px; object-fit: cover;
    border-radius: var(--r-sm);
    box-shadow: var(--shadow-sm);
}
.dest-name {
    font-family: var(--font-serif);
    font-size: 1.3rem;
    font-weight: 600;
    color: var(--text-main);
    line-height: 1.35;
    margin-bottom: 6px;
    overflow: visible;
}
.match-badge {
    background: rgba(158, 68, 41, 0.08); color: var(--primary);
    border: 1px solid rgba(158, 68, 41, 0.2);
    padding: 3px 10px; border-radius: 12px; font-weight: 600; font-size: 0.78rem;
    letter-spacing: 0.02em;
}
.cluster-tag {
    font-size: 0.73rem; font-weight: 500; padding: 3px 10px;
    border-radius: 12px; display: inline-block;
    letter-spacing: 0.02em;
    background: #F2EFE9; color: var(--text-2);
    border: 1px solid var(--border);
}
.dest-meta {
    display: flex; align-items: center; gap: 8px; flex-wrap: wrap;
    font-size: 0.85rem; color: var(--text-2); margin-bottom: 8px;
    line-height: 1.4;
}
.meta-sep {
    color: #C0B8AA; margin: 0 2px;
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


def safe_image_tag(url, class_name="dest-card-img", fallback_icon="🏝️"):
    url_str = str(url).strip() if (url and not pd.isna(url)) else ""
    if url_str and (url_str.startswith("http://") or url_str.startswith("https://")):
        return f'<img src="{url_str}" class="{class_name}" alt="Destination" onerror="this.onerror=null; this.src=\'https://images.unsplash.com/photo-1528127269322-539801943592?auto=format&fit=crop&w=800&q=80\';"/>'
    return f'<div class="{class_name}" style="background:var(--cream-dk); display:flex; align-items:center; justify-content:center; font-size:3rem; min-height:160px;">{fallback_icon}</div>'


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
def run_offline_eval(_result_df, _vectorizer, _tfidf_matrix, community_key):
    return offline_evaluation(_result_df, _vectorizer, _tfidf_matrix, community=dict(community_key))


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

h_col1, h_col2 = st.columns([3, 1], vertical_alignment="center")

with h_col1:
    st.markdown("""
    <div style="padding: 6px 0; overflow: visible;">
        <div style="font-family:var(--font-serif); font-size:1.45rem; font-weight:600; color:var(--text-main); line-height:1.45; overflow:visible;">
            Hệ thống Trợ giúp Ra quyết định Du lịch
        </div>
        <div style="font-size:0.75rem; font-weight:700; text-transform:uppercase; letter-spacing:0.12em; color:var(--gold); line-height:1.4; margin-top:2px;">
            ĐỒ ÁN MÔN DSS · K-MEANS · TOPSIS · CONTENT-BASED FILTERING
        </div>
    </div>
    """, unsafe_allow_html=True)

with h_col2:
    if not st.session_state["is_admin_logged_in"]:
        if st.button("ĐĂNG NHẬP", key="top_admin_btn"):
            st.session_state["show_top_admin_login"] = not st.session_state.get("show_top_admin_login", False)
            st.rerun()
    else:
        if st.button("ĐĂNG XUẤT", key="top_logout_btn"):
            st.session_state["is_admin_logged_in"] = False
            st.session_state["active_view"] = "SURVEY"
            st.rerun()

# Modal / Card Đăng nhập Admin xuất hiện ngay dưới Header khi bấm nút
if not st.session_state["is_admin_logged_in"] and st.session_state.get("show_top_admin_login", False):
    with st.container(border=True):
        st.markdown("<div class='section-title' style='font-size:1.25rem !important; margin-bottom:16px !important;'>Đăng nhập Quản trị viên</div>", unsafe_allow_html=True)
        tu1, tu2 = st.columns(2)
        with tu1:
            top_u = st.text_input("Tài khoản", key="top_u_input")
        with tu2:
            top_p = st.text_input("Mật khẩu", type="password", key="top_p_input")

        if st.button("XÁC NHẬN", type="primary", key="top_sub_btn"):
            if top_u == "admin" and top_p == "admin123":
                st.session_state["is_admin_logged_in"] = True
                st.session_state["show_top_admin_login"] = False
                st.session_state["active_view"] = "ADMIN"
                st.rerun()
            else:
                st.error("Tài khoản hoặc mật khẩu không chính xác.")


# Thanh chuyển View nếu Admin đã đăng nhập hoặc đã có kết quả
if st.session_state["is_admin_logged_in"]:
    st.markdown("<div style='margin-bottom:16px;'>", unsafe_allow_html=True)
    v1, v2, v3 = st.columns(3)
    with v1:
        if st.button("📝 KHẢO SÁT NHU CẦU", use_container_width=True):
            st.session_state["active_view"] = "SURVEY"
            st.rerun()
    with v2:
        if st.button("🧭 GỢI Ý THEO 3 MIỀN", use_container_width=True):
            st.session_state["active_view"] = "RECOMMENDATIONS"
            st.rerun()
    with v3:
        if st.button("📊 QUẢN TRỊ ADMIN", type="primary", use_container_width=True):
            st.session_state["active_view"] = "ADMIN"
            st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)
elif st.session_state.get("has_run", False):
    st.markdown("<div style='margin-bottom:16px;'>", unsafe_allow_html=True)
    v1, v2 = st.columns(2)
    with v1:
        if st.button("📝 SỬA KHẢO SÁT NHU CẦU", use_container_width=True):
            st.session_state["active_view"] = "SURVEY"
            st.rerun()
    with v2:
        if st.button("🧭 XEM GỢI Ý THEO 3 MIỀN", type="primary", use_container_width=True):
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
        <div class="hero-title">Thiết kế chuyến đi của riêng bạn</div>
        <div class="hero-description">
            Hãy chia sẻ nguyện vọng và sở thích để hệ thống gợi ý những điểm đến hoàn hảo nhất dành riêng cho bạn.
        </div>
    </div>
    """, unsafe_allow_html=True)

    with st.container(border=True):
        st.markdown('<div class="section-title">Khảo sát Thông tin & Ràng buộc</div>', unsafe_allow_html=True)

        # 1. Ô nhập từ khóa / câu văn tự do
        style_input = st.text_input(
            "PHONG CÁCH & YÊU CẦU DU LỊCH (NHẬP TỪ KHÓA HOẶC CÂU VĂN TỰ DO)",
            value="",
            placeholder="Ví dụ: Tôi muốn đi đến địa điểm có phong cảnh thiên nhiên xanh mát...",
            help="Nhập sở thích hoặc yêu cầu cụ thể của bạn về chuyến đi."
        )

        parsed_kws, boosts = extract_intent_keywords(style_input)
        if parsed_kws:
            st.caption(f"✓ Đã nhận diện các từ khóa: {', '.join(parsed_kws)} - Hệ thống sẽ tự động ưu tiên.")

        st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)

        # 2. Hàng thông tin ràng buộc chuyến đi (Ngân sách & Số ngày & Số người)
        rc1, rc2, rc3 = st.columns(3)
        with rc1:
            num_people_input = st.number_input(
                "SỐ NGƯỜI ĐI",
                min_value=1,
                max_value=50,
                value=1,
                step=1,
                help="Nhập số lượng người tham gia chuyến đi."
            )
        with rc2:
            budget_input = st.number_input(
                "NGÂN SÁCH / NGƯỜI (VNĐ)",
                min_value=200000,
                max_value=50000000,
                value=3000000,
                step=500000,
                help="Nhập chi phí dự kiến cho MỘT NGƯỜI. Hệ thống sẽ tự động nhân với số người."
            )
        with rc3:
            duration = st.number_input(
                "SỐ NGÀY LƯU TRÚ",
                min_value=0.5,
                max_value=14.0,
                value=2.0,
                step=0.5,
                help="Số ngày bạn có thể dành cho chuyến tham quan."
            )

        st.markdown("<hr style='margin: 20px 0;'>", unsafe_allow_html=True)

        if st.button("🔍 PHÂN TÍCH & TÌM KIẾM GỢI Ý", type="primary", use_container_width=True):
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
        st.warning("⚠️ Bạn chưa nhập thông tin khảo sát. Vui lòng hoàn thành khảo sát nhu cầu trước.")
        if st.button("➡️ CHUYỂN TỚI KHẢO SÁT NHU CẦU"):
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
        st.markdown(f"""
        <div class="hero">
            <div class="hero-content">
                <div class="hero-title">Gợi ý Danh sách Địa điểm Du lịch theo 3 Miền</div>
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
            preferred_duration=(max(0.5, duration - 1), duration + 1),
            keyword_query=style,
            tfidf_model=vectorizer,
            tfidf_matrix=tfidf_matrix,
            community=community,
            hotel_df=hotel_df
        )

        # Region Tabs
        st.markdown('<div class="section-title">Gợi ý điểm đến hàng đầu theo 3 Miền</div>', unsafe_allow_html=True)
        st.info("💡 **Lưu ý về Ngân sách:** Độ phù hợp ngân sách (Budget Fit) được hệ thống tính toán bao gồm **giá vé vào cổng** kết hợp với **trung bình giá phòng khách sạn** lân cận.")
        t_mb, t_mt, t_mn = st.tabs(["MIỀN BẮC", "MIỀN TRUNG", "MIỀN NAM"])

        def format_sub_rating(val):
            if pd.notna(val):
                try:
                    n = float(val)
                    if not np.isnan(n) and n > 0:
                        return f" (★ {n:.1f})"
                except (ValueError, TypeError):
                    pass
            return ""

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
                rating_part = f"<span>★ {r_num:.1f}</span><span class='meta-sep'>•</span>" if (pd.notna(r_num) and r_num > 0) else ""
                province_part = f"<span>{row.get('province', '')}</span>" if row.get('province') else ""
                price_str = format_price_range(row.get('cost_min'), row.get('cost_max'))
                price_part = f"<span class='meta-sep'>•</span><span title='Giá vé vào cổng (Hệ thống đã tự động cộng thêm chi phí khách sạn khi đánh giá mức độ phù hợp ngân sách)'>Vé vào cổng: {price_str}</span>" if price_str else ""

                col_img, col_info = st.columns([1, 2])
                with col_img:
                    st.markdown(banner_html, unsafe_allow_html=True)
                with col_info:
                    card_html = f"""<div class="dest-name">{row['destination_name'].strip()}</div>
<div class="dest-meta">
<span class="match-badge">{score:.0f}% Phù hợp</span>
{rating_part}
{province_part}
{price_part}
</div>
<p style="font-size:0.86rem; color:var(--text-2); margin-bottom:8px;">{str(row.get('description',''))[:150]}...</p>
<span class="cluster-tag">{cluster_clean}</span>"""
                    st.markdown(card_html, unsafe_allow_html=True)

                    with st.expander(f"Xem chi tiết dịch vụ & lý do gợi ý cho {row['destination_name'].strip()}"):
                        # Log implicit view: nội dung expander chạy lại ở MỖI lần rerun nên phải
                        # khử trùng theo phiên, nếu không mỗi lần bấm sao/radio đều cộng lại lượt xem.
                        _logged = st.session_state.setdefault("_logged_views", set())
                        if row["destination_id"] not in _logged:
                            log_implicit_view(row["destination_id"], row["destination_name"].strip(), region_name)
                            _logged.add(row["destination_id"])

                        reasons = explain_top_criteria(
                            row, budget=budget_per_person, keyword_query=style,
                            feedback=fb_counts.get(row["destination_id"]),
                        )
                        if reasons:
                            st.markdown("**VÌ SAO LỰA CHỌN NÀY PHÙ HỢP?**")
                            for r in reasons:
                                st.markdown(f"- {r}")

                        c_hotel, c_rest = st.columns(2)
                        with c_hotel:
                            st.markdown("<b>Khách sạn lân cận:</b>", unsafe_allow_html=True)
                            hotels = hotel_df[hotel_df["destination_id"] == row["destination_id"]].head(2)
                            if hotels.empty:
                                st.caption("Đang cập nhật khách sạn...")
                            for _, h in hotels.iterrows():
                                h_name = clean_title_icon(h.get('hotel_name', ''))
                                h_rate = format_sub_rating(h.get('rating_num'))
                                st.markdown(f"- {h_name}{h_rate}")
                        with c_rest:
                            st.markdown("<b>Quán ăn gợi ý:</b>", unsafe_allow_html=True)
                            rests = rest_df[rest_df["destination_id"] == row["destination_id"]].head(2)
                            if rests.empty:
                                st.caption("Đang cập nhật quán ăn...")
                            for _, r in rests.iterrows():
                                r_name = clean_title_icon(r.get('restaurant_name', ''))
                                r_rate = format_sub_rating(r.get('rating_num'))
                                st.markdown(f"- {r_name}{r_rate}")

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
            st.markdown('<div class="section-title">Kết quả này có hữu ích không?</div>', unsafe_allow_html=True)
            st.markdown('<div style="font-size:0.9rem; color:var(--text-muted); margin-bottom:16px;">Đánh giá của bạn giúp chúng tôi cải thiện thuật toán gợi ý.</div>', unsafe_allow_html=True)

            st.markdown("<b style='font-size:0.80rem; text-transform:uppercase;'>MỨC ĐỘ HÀI LÒNG:</b>", unsafe_allow_html=True)
            stars_choice = st.radio(
                "Mức độ hài lòng",
                [1, 2, 3, 4, 5],
                index=None,
                format_func=lambda x: "⭐" * x + f" ({x} sao)",
                horizontal=True,
                label_visibility="collapsed"
            )

            st.markdown("<br><b style='font-size:0.80rem; text-transform:uppercase;'>ĐÁNH GIÁ SỰ PHÙ HỢP CỦA CÁC ĐỊA ĐIỂM TOP ĐẦU:</b>", unsafe_allow_html=True)

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
                "GÓP Ý THÊM (KHÔNG BẮT BUỘC)",
                placeholder="Nhập ý kiến nhận xét của bạn về các địa điểm gợi ý..."
            )

            if st.button("GỬI ĐÁNH GIÁ", type="primary"):
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
                    st.success("✅ Cảm ơn bạn! Đánh giá đã được lưu thành công vào hệ thống.")


# =========================================================
# GIAO DIỆN 3: QUẢN TRỊ ADMIN - PHẢN HỒI NGƯỜI DÙNG (IMAGE 2 STYLE)
# =========================================================
elif st.session_state["active_view"] == "ADMIN":
    if not st.session_state.get("is_admin_logged_in", False):
        st.error("⛔ Bạn chưa đăng nhập quyền Quản trị viên. Vui lòng bấm nút '🔑 ĐĂNG NHẬP ADMIN' ở góc trên bên phải.")
        st.stop()

    st.markdown("""
    <div class="hero">
        <div class="hero-content">
            <span class="hero-eyebrow">KHU VỰC QUẢN TRỊ VIÊN</span>
            <div class="hero-title">PHẢN HỒI NGƯỜI DÙNG & TƯƠNG TÁC THỰC TẾ</div>
            <div class="hero-description">
                Tổng hợp Explicit Feedback (đánh giá tường minh) và Implicit Feedback (hành vi ngầm) thu thập từ hệ thống.
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    stats = get_feedback_stats()

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
            <div class="kpi-value">{stats['avg_stars']} / 5 SAO</div>
            <div class="kpi-label">ĐIỂM TRUNG BÌNH</div>
        </div>
        """, unsafe_allow_html=True)
    with m3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-value">{stats['total_views']} LƯỢT</div>
            <div class="kpi-label">TƯƠNG TÁC NGẦM (IMPLICIT VIEWS)</div>
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
        eval_tbl, n_scn = run_offline_eval(result_df, vectorizer, tfidf_matrix, tuple(sorted(_comm.items())))
        st.dataframe(eval_tbl, use_container_width=True, hide_index=True)
        st.caption(
            f"Mô phỏng {n_scn} kịch bản (ngân sách × số ngày × chủ đề). Một địa điểm 'phù hợp' khi thỏa các ràng buộc "
            "khách nêu: chi phí tối đa ≤ ngân sách, thời lượng lệch ≤ 1 ngày, và đúng chủ đề theo tên/từ khóa gán tay "
            "(độc lập với TF-IDF của bộ xếp hạng). Dòng 'Giới hạn trên' là mức tối đa có thể đạt. "
            "Đây là đánh giá theo yêu cầu đã nêu, không thay thế đánh giá của người dùng thật."
        )

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
                    "Sao": "⭐" * r["stars"],
                    "Phù hợp": ", ".join(r["fit_items"]) if r["fit_items"] else "—",
                    "Chưa phù hợp": ", ".join(r["unfit_items"]) if r["unfit_items"] else "—",
                    "Bình luận": r["comment"] if r["comment"] else "—"
                })
            df_exp = pd.DataFrame(formatted_rows)
            st.dataframe(df_exp, use_container_width=True, hide_index=True)
        else:
            st.info("Chưa có lượt đánh giá tường minh nào.")