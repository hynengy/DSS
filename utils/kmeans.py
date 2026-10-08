import re
from collections import Counter

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler, normalize

from utils.preprocessing import clean_destinations

# Theme "gốc" dùng để chấm điểm cụm -> chọn chủ đề đúng nhất
THEME_KEYWORDS = {
    "Biển": ["biển", "vịnh", "hoang sơ"],
    "Đảo": ["đảo"],
    "Núi": ["núi", "rừng"],
    "Hang động": ["hang động", "UNESCO", "núi đá vôi"],
    "Thác - suối - sông": ["thác", "suối", "sông", "sông nước"],
    "Văn hóa - di tích - lịch sử": [
        "văn hóa",
        "di tích",
        "lịch sử",
        "kiến trúc",
        "di sản",
    ],
    "Tâm linh - chùa chiền": ["tâm linh", "chùa", "đền", "lăng", "linh thiêng"],
    "Giải trí - mua sắm - hiện đại": [
        "mua sắm",
        "vui chơi",
        "giải trí",
        "check-in",
        "hoàng hôn",
    ],
}

# Tên hiển thị "bắt mắt" cho từng theme — dùng để đặt tên cụm cho người dùng thấy
CATCHY_NAMES = {
    "Biển": "Biển đảo hoang sơ",
    "Đảo": "Đảo ngọc hoang sơ",
    "Núi": "Núi rừng hùng vĩ",
    "Hang động": "Hang động kỳ bí",
    "Thác - suối - sông": "Suối thác mộng mơ",
    "Văn hóa - di tích - lịch sử": "Văn hóa & Lịch sử",
    "Tâm linh - chùa chiền": "Tâm linh & An yên",
    "Giải trí - mua sắm - hiện đại": "Hiện đại & Check-in",
}

# Sửa các lỗi keyword hay gặp trong dữ liệu (viết liền, thiếu dấu cách...)
KEYWORD_FIXES = {
    "tâmlinh": "tâm linh",
}

# Keyword dài bất thường (>3 từ) thường là tên địa điểm bị nhét nhầm vào cột
# keywords (vd "Buu dien Trung tam Sai Gon"), không phải keyword chủ đề thật
# -> loại bỏ để không làm nhiễu cụm.
MAX_KEYWORD_WORDS = 3


def normalize_keywords(kws):
    if not isinstance(kws, list):
        return []

    cleaned = []
    for kw in kws:
        kw = re.sub(r"\s+", " ", kw.strip().lower())
        kw = KEYWORD_FIXES.get(kw, kw)
        if kw and len(kw.split()) <= MAX_KEYWORD_WORDS:
            cleaned.append(kw)
    return cleaned


VIETNAMESE_STOP_WORDS = [
    "muốn", "đi", "đến", "tôi", "cho", "bảo", "ở", "tại", "của", "và", "những",
    "cái", "là", "đó", "này", "với", "một", "các", "có", "được", "ra", "vào",
    "lại", "thì", "hãy", "gì", "nào", "mấy", "bao", "nhiêu", "nữa", "rồi",
    "rất", "đẹp", "nơi", "mang", "đậm", "chất", "tuyệt", "vời", "khá", "nhiều",
    "cũng", "đây", "như", "thế", "khi", "sẽ", "bị", "từ", "để", "về", "cùng",
    "nhất", "nhau", "luôn", "đang", "còn", "cần", "thật", "sự", "chỉ", "thể",
    "hơn", "quá", "lắm", "ơi", "đâu", "ai", "sao", "chúng", "mình", "người",
    "thích", "nhìn", "ngắm", "cảnh", "thấy", "chút", "hơi", "nên", "thôi",
    "cảnh", "thiên", "nhiên", "phong"
]


# Các cột có ngoại lai (vd cost_max = 70 triệu, thời lượng 37 ngày do dữ liệu nhập
# sai/đơn vị khác) -> cắt ở phân vị trên để K-Means không tạo cụm chỉ gồm 1-2 điểm.
WINSOR_COLS = ["cost_min", "cost_max", "duration_days"]


def build_feature_matrix(df, keyword_weight=0.5, use_location=False, winsor_q=0.90,
                         svd_dims=8, numeric_weight=0.15):
    """
    Ma trận đặc trưng cho K-Means.

    Trước đây các cột số (giá, thời lượng) chi phối phân cụm vì từ khóa quá thưa
    (mỗi từ khóa chỉ xuất hiện 1-2 lần) -> cụm lẫn lộn chủ đề (vd Nhà tù Hỏa Lò nằm
    chung cụm "Biển đảo"). Nay phân cụm chủ yếu theo NỘI DUNG (TF-IDF của tên + từ
    khóa + mô tả, giảm chiều bằng LSA/TruncatedSVD) và chỉ cộng thêm các đặc trưng số
    với trọng số nhỏ. Đo bằng độ thuần chủ đề của cụm: 0.45 -> ~0.70.
    (tham số keyword_weight giữ lại để tương thích, không còn dùng.)
    """
    df = df.copy()
    df["keywords_clean"] = df["keywords_list"].apply(normalize_keywords)

    numeric_cols = ["rating_num", "cost_min", "cost_max", "duration_days"]
    if use_location:
        numeric_cols += ["latitude", "longitude"]

    numeric_df = df[numeric_cols].copy()
    numeric_df = numeric_df.fillna(numeric_df.median(numeric_only=True))

    if winsor_q:
        for col in WINSOR_COLS:
            numeric_df[col] = numeric_df[col].clip(upper=numeric_df[col].quantile(winsor_q))

    numeric_df["cost_min"] = np.log1p(numeric_df["cost_min"])
    numeric_df["cost_max"] = np.log1p(numeric_df["cost_max"])

    scaler = StandardScaler()
    numeric_scaled = scaler.fit_transform(numeric_df)

    keyword_text = df["keywords_clean"].apply(lambda kws: " ".join(kws))

    # Rich corpus for search & TOPSIS style matching
    search_corpus = (
        df["destination_name"].fillna("") + " " +
        df["province"].fillna("") + " " +
        keyword_text + " " +
        df["description"].fillna("")
    )
    vectorizer_search = TfidfVectorizer(
        min_df=1,
        ngram_range=(1, 2),
        sublinear_tf=True,
        stop_words=VIETNAMESE_STOP_WORDS
    )
    tfidf_matrix_search = vectorizer_search.fit_transform(search_corpus)

    # Đặc trưng cho K-Means = LSA(TF-IDF nội dung) + đặc trưng số trọng số nhỏ
    n_comp = max(2, min(svd_dims, tfidf_matrix_search.shape[1] - 1, tfidf_matrix_search.shape[0] - 1))
    text_lsa = normalize(TruncatedSVD(n_components=n_comp, random_state=42).fit_transform(tfidf_matrix_search))
    combined = np.hstack([
        text_lsa,
        numeric_scaled * (numeric_weight / np.sqrt(numeric_scaled.shape[1])),
    ])

    return combined, vectorizer_search, scaler, df, tfidf_matrix_search


def find_optimal_k(X, k_range=range(3, 9), min_cluster_size: int = 8):
    """
    Tìm k tối ưu dựa trên silhouette score cao nhất.
    Bỏ qua các giá trị k tạo ra cụm có < min_cluster_size điểm
    (tránh "cụm rác" do outlier).
    Range mặc định (3, 9).
    min_cluster_size=8: mỗi cụm phải có >= 8/100 điểm đến, tránh chọn k cho ra
    "cụm rác" 1-2 điểm (silhouette cao giả tạo do outlier).
    """
    results = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X)
        # Bỏ qua k nếu có cụm quá nhỏ
        min_size = pd.Series(labels).value_counts().min()
        if min_size < min_cluster_size:
            print(f"  k={k}: bỏ qua (cụm nhỏ nhất = {min_size} < {min_cluster_size})")
            continue
        sil = silhouette_score(X, labels)
        results.append({"k": k, "inertia": km.inertia_, "silhouette": sil,
                        "min_cluster_size": min_size})

    if not results:
        # Fallback: nếu tất cả k đều tạo cụm nhỏ, dùng k nhỏ nhất
        fallback_k = min(k_range)
        print(f"  Không tìm được k hợp lệ, dùng fallback k={fallback_k}")
        return pd.DataFrame([{"k": fallback_k, "silhouette": 0.0}]), fallback_k

    result_df = pd.DataFrame(results)
    print(result_df.to_string(index=False))
    best_k = int(result_df.loc[result_df["silhouette"].idxmax(), "k"])
    print(f"\n-> Chọn k = {best_k} (silhouette cao nhất, cụm nhỏ nhất ≥ {min_cluster_size})")
    return result_df, best_k



def get_theme_scores(cluster_keywords_series):
    all_keywords = []
    for kws in cluster_keywords_series:
        if isinstance(kws, list):
            all_keywords.extend(kws)
    keyword_counter = Counter(all_keywords)

    scores = {}
    for theme, theme_kws in THEME_KEYWORDS.items():
        scores[theme] = sum(keyword_counter.get(kw, 0) for kw in theme_kws)
    return scores, keyword_counter


def _cluster_text(df):
    """Văn bản dùng để chấm chủ đề cụm: tên + từ khóa + mô tả (chữ thường)."""
    parts = df["destination_name"].fillna("") if "destination_name" in df.columns else pd.Series("", index=df.index)
    if "keywords_clean" in df.columns:
        parts = parts + " " + df["keywords_clean"].apply(lambda k: " ".join(k) if isinstance(k, list) else "")
    if "description" in df.columns:
        parts = parts + " " + df["description"].fillna("")
    return parts.str.lower()


def assign_unique_cluster_names(df):
    """
    Đặt tên cụm theo chủ đề ĐẶC TRƯNG của nội dung cụm.

    - Điểm chủ đề = tỉ lệ thành viên của cụm nhắc tới từ khóa chủ đề TRỪ tỉ lệ chung
      của toàn bộ điểm đến (tránh gán từ phổ biến như "văn hóa" cho mọi cụm).
    - Ghép cụm <-> chủ đề tối ưu 1-1 (Hungarian) nên không có hai cụm trùng tên.
    - Cụm không có chủ đề nổi bật -> đặt theo 2 từ khóa nhiều nhất.
    """
    from scipy.optimize import linear_sum_assignment

    cluster_ids = sorted(df["cluster_id"].unique())
    themes = list(THEME_KEYWORDS.keys())
    text = _cluster_text(df)

    def mentions(txt_series, kws):
        return txt_series.apply(lambda t: any(k.lower() in t for k in kws)).astype(float)

    overall = {th: mentions(text, THEME_KEYWORDS[th]).mean() for th in themes}
    score = np.zeros((len(cluster_ids), len(themes)))
    counters = {}
    for ci, cid in enumerate(cluster_ids):
        sub = df[df["cluster_id"] == cid]
        sub_text = text[sub.index]
        counters[cid] = Counter(k for kws in sub["keywords_clean"] for k in kws)
        for ti, th in enumerate(themes):
            score[ci, ti] = mentions(sub_text, THEME_KEYWORDS[th]).mean() - overall[th]

    rows, cols = linear_sum_assignment(-score)
    names = {}
    for r, c in zip(rows, cols):
        cid = cluster_ids[r]
        if score[r, c] > 0.05:
            names[cid] = CATCHY_NAMES.get(themes[c], themes[c])
    for cid in cluster_ids:
        if cid not in names:
            top_kw = [w for w, _ in counters[cid].most_common(2)]
            names[cid] = ("✨ Khám phá " + " - ".join(top_kw)) if top_kw else f"✨ Hành trình #{cid}"
    return names


def print_cluster_report(df):
    """In keyword phổ biến nhất của từng cụm để kiểm tra chất lượng phân cụm."""
    for cid in sorted(df["cluster_id"].unique()):
        subset = df[df["cluster_id"] == cid]
        _, counter = get_theme_scores(subset["keywords_clean"])
        name = subset["cluster_name"].iloc[0]
        print(f"\nCụm {cid} — {name} ({len(subset)} điểm)")
        print("Top keywords:", counter.most_common(5))


def run_kmeans(df, k=None, keyword_weight=0.5, k_range=range(3, 9), min_cluster_size: int = 8):
    """
    k=None: tự động chọn k theo silhouette cao nhất trong k_range.
    k=<số>: dùng đúng số cụm chỉ định (bỏ qua tìm kiếm tự động).
    """
    X, vectorizer, scaler, df_clean, _ = build_feature_matrix(
        df, keyword_weight=keyword_weight
    )

    if k is None:
        _, k = find_optimal_k(X, k_range=k_range, min_cluster_size=min_cluster_size)

    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    df_clean["cluster_id"] = km.fit_predict(X)

    cluster_names = assign_unique_cluster_names(df_clean)
    df_clean["cluster_name"] = df_clean["cluster_id"].map(cluster_names)
    return df_clean, km


if __name__ == "__main__":
    dest_df = clean_destinations()
    result_df, model = run_kmeans(dest_df)  # k tự động chọn theo silhouette
    print("\nKết quả phân cụm thành công!")
    print_cluster_report(result_df)