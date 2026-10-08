"""
TOPSIS (Technique for Order Preference by Similarity to Ideal Solution)

Dùng để thực hiện 2 bước trong chu trình DSS:
    3. Dự đoán mức độ phù hợp (suitability score) của từng điểm đến với hồ sơ
       sở thích / ràng buộc của người dùng (ngân sách, đánh giá, thời lượng,
       phong cách yêu thích).
    4. Xếp hạng (ranking) các điểm đến theo điểm phù hợp đó.

Ý tưởng TOPSIS: điểm đến "tốt nhất" là điểm đến gần với giải pháp lý tưởng
(ideal best - đạt điểm tối ưu ở mọi tiêu chí) và xa giải pháp phản lý tưởng
(ideal worst - tệ nhất ở mọi tiêu chí) nhất.

Thay đổi so với phiên bản trước:
  - Thay tiêu chí "cost_avg" (cost → càng rẻ càng tốt) bằng "budget_fit"
    (benefit → khoảng cách tới ngân sách user, cao hơn = phù hợp hơn).
  - Tích hợp Cosine Similarity từ TF-IDF vào "style_match" khi có keyword query.
  - Thêm tham số keyword_query / tfidf_model / tfidf_matrix vào build_decision_matrix.

Bổ sung (tích hợp học máy vào bước 3):
  - "cluster_fit": độ gần giữa câu mô tả sở thích của khách và TÂM CỤM K-Means của
    điểm đến (cosine TF-IDF) -> K-Means thực sự tham gia vào xếp hạng, đồng thời mở
    rộng tập ứng viên sang các điểm đến cùng cụm (cluster hypothesis).
  - "community_score": mức chấp nhận của điểm đến học từ phản hồi người dùng trước
    (Beta-smoothing), xem utils/learning.py. Chưa có phản hồi -> trung lập 0.5.
  - Xuất thêm cột "topsis_raw" (điểm TOPSIS gốc, chưa co giãn 60-98%) và các cột
    "f_<tiêu chí>" để ghi lại đặc trưng khi người dùng đánh giá (dùng huấn luyện).
"""

import numpy as np
import pandas as pd

# Trọng số & hướng tối ưu mặc định cho từng tiêu chí.
# "benefit": càng cao càng tốt | "cost": càng thấp càng tốt
DEFAULT_CRITERIA = {
    "rating_num":      {"weight": 0.35, "type": "benefit"},
    "budget_fit":      {"weight": 0.30, "type": "benefit"},  # thay cost_avg
    "duration_fit":    {"weight": 0.20, "type": "benefit"},
    "style_match":     {"weight": 0.15, "type": "benefit"},
    "cluster_fit":     {"weight": 0.00, "type": "benefit"},  # K-Means -> xếp hạng
    "community_score": {"weight": 0.00, "type": "benefit"},  # học từ phản hồi
}

# Cụm có độ gần >= ngưỡng này (so với cụm gần nhất) vẫn được giữ làm ứng viên
CLUSTER_KEEP_THRESHOLD = 0.75


def recommended_weights(has_query: bool) -> dict:
    """
    Trọng số khởi điểm (chưa học từ phản hồi) cho 6 tiêu chí, đã chuẩn hoá tổng = 1.
    Có câu mô tả sở thích -> ưu tiên độ khớp nội dung (style_match) và cụm K-Means;
    không có -> chỉ còn rating / ngân sách / thời lượng (hai tiêu chí kia là hằng số).
    """
    if has_query:
        w = {"rating_num": 0.15, "budget_fit": 0.13, "duration_fit": 0.08,
             "style_match": 0.40, "cluster_fit": 0.16, "community_score": 0.08}
    else:
        w = {"rating_num": 0.32, "budget_fit": 0.30, "duration_fit": 0.20,
             "style_match": 0.04, "cluster_fit": 0.04, "community_score": 0.10}
    tot = sum(w.values())
    return {k: v / tot for k, v in w.items()}


def _vector_normalize(col: pd.Series) -> pd.Series:
    denom = np.sqrt((col.astype(float) ** 2).sum())
    if denom == 0 or np.isnan(denom):
        return col.astype(float) * 0.0
    return col.astype(float) / denom


def topsis_score(matrix: pd.DataFrame, criteria: dict) -> pd.Series:
    """
    matrix: DataFrame, mỗi cột là 1 tiêu chí, mỗi dòng là 1 phương án (điểm đến)
    criteria: dict {ten_cot: {"weight": float, "type": "benefit"|"cost"}}

    Trả về: Series điểm phù hợp đã chuẩn hoá 0..1 (1 = gần lý tưởng nhất),
    cùng index với matrix.
    """
    cols = [c for c in criteria if c in matrix.columns]
    if not cols or matrix.empty:
        return pd.Series(0.5, index=matrix.index)

    # B1: chuẩn hoá vector từng cột
    norm = matrix[cols].apply(_vector_normalize, axis=0)

    # B2: nhân trọng số
    weights = np.array([criteria[c]["weight"] for c in cols])
    weights = weights / weights.sum() if weights.sum() > 0 else weights
    weighted = norm * weights

    # B3: xác định giải pháp lý tưởng tốt nhất (A+) / tệ nhất (A-)
    ideal_best, ideal_worst = {}, {}
    for c in cols:
        if criteria[c]["type"] == "cost":
            ideal_best[c]  = weighted[c].min()
            ideal_worst[c] = weighted[c].max()
        else:
            ideal_best[c]  = weighted[c].max()
            ideal_worst[c] = weighted[c].min()

    # B4: khoảng cách Euclid đến A+ và A-
    dist_best  = np.sqrt(sum((weighted[c] - ideal_best[c])  ** 2 for c in cols))
    dist_worst = np.sqrt(sum((weighted[c] - ideal_worst[c]) ** 2 for c in cols))

    # B5: độ gần tương đối với giải pháp lý tưởng (closeness coefficient)
    denom = dist_best + dist_worst
    score = np.where(denom == 0, 0.5, dist_worst / denom.replace(0, np.nan))
    score = pd.Series(score, index=matrix.index).fillna(0.5)
    return score.clip(0, 1)


SYNONYM_MAP = {
    "vịnh": "vịnh biển đảo bãi biển",
    "biển": "biển bãi tắm đảo bãi biển vịnh",
    "đảo": "đảo quần đảo biển bãi tắm",
    "núi": "núi đồi cao nguyên trekking leo núi đèo",
    "thác": "thác suối sông",
    "lịch sử": "lịch sử di tích cổ kính di sản bảo tàng",
    "chùa": "chùa đền tâm linh miếu thiền viện",
    "nghỉ dưỡng": "nghỉ dưỡng resort spa thư giãn",
    "phượt": "phượt trekking cắm trại",
}


def expand_query_terms(text: str) -> str:
    if not text or not str(text).strip():
        return ""
    text_str = str(text).strip()
    text_lower = text_str.lower()
    additions = []
    for term, exp in SYNONYM_MAP.items():
        if term in text_lower:
            additions.append(exp)
    if additions:
        return text_str + " " + " ".join(additions)
    return text_str


def cluster_affinity(df, keyword_query, tfidf_model, tfidf_matrix):
    """
    Độ gần (0..1) giữa câu mô tả sở thích và TÂM CỤM K-Means của từng điểm đến.
    Trả về None nếu không tính được (không có query / lệch kích thước / thiếu cluster_id).
    """
    if not keyword_query or tfidf_model is None or tfidf_matrix is None:
        return None
    if "cluster_id" not in df.columns or tfidf_matrix.shape[0] != len(df):
        return None
    from sklearn.metrics.pairwise import cosine_similarity as _cos_sim
    q_vec = tfidf_model.transform([expand_query_terms(keyword_query)])
    ids = df["cluster_id"].to_numpy()
    aff = {}
    for cid in np.unique(ids):
        centroid = np.asarray(tfidf_matrix[ids == cid].mean(axis=0))
        aff[cid] = float(_cos_sim(q_vec, centroid)[0, 0])
    s = pd.Series(ids, index=df.index).map(aff).astype(float)
    mx = s.max()
    return (s / mx) if mx > 0 else None


def build_decision_matrix(
    df: pd.DataFrame,
    preferred_duration=None,
    selected_clusters=None,
    budget: float = None,
    keyword_query: str = None,
    tfidf_model=None,
    tfidf_matrix=None,
    community: dict = None,
) -> pd.DataFrame:
    """
    Xây ma trận quyết định (decision matrix) từ dữ liệu điểm đến đã lọc,
    dựa trên hồ sơ / lựa chọn hiện tại của người dùng.
    """
    m = pd.DataFrame(index=df.index)

    # ── rating_num ──────────────────────────────────────────
    r_val = df["rating_num"].fillna(df["rating_num"].median())
    r_min, r_max = r_val.min(), r_val.max()
    if r_max > r_min:
        m["rating_num"] = (r_val - r_min) / (r_max - r_min)
    else:
        m["rating_num"] = 1.0

    # ── budget_fit: khoảng cách tương đối tới ngân sách ────
    # Giá trị ∈ [0, 1]: 1 = miễn phí, 0 = vượt ngân sách
    avg_cost = df[["cost_min", "cost_max"]].mean(axis=1).fillna(0)
    if budget and budget > 0:
        m["budget_fit"] = ((budget - avg_cost) / budget).clip(lower=0.0, upper=1.0)
    else:
        m["budget_fit"] = 1.0  # Không có budget → không phạt

    # ── duration_fit ─────────────────────────────────────────
    if preferred_duration:
        lo, hi = preferred_duration
        mid = (lo + hi) / 2 if hi < 999 else lo + 1
        m["duration_fit"] = -(df["duration_days"].fillna(mid) - mid).abs()
    else:
        m["duration_fit"] = 0.0

    # ── style_match: TF-IDF cosine > cluster filter > uniform ─
    if keyword_query and tfidf_model is not None and tfidf_matrix is not None:
        from sklearn.metrics.pairwise import cosine_similarity as _cos_sim
        query_to_use = expand_query_terms(keyword_query)
        user_vec = tfidf_model.transform([query_to_use])
        sims = _cos_sim(user_vec, tfidf_matrix).flatten()
        if len(sims) == len(df):
            m["style_match"] = sims
        else:
            all_sims = pd.Series(sims, index=range(len(sims)))
            m["style_match"] = df.reset_index(drop=True).index.map(
                lambda i: all_sims.iloc[i] if i < len(all_sims) else 0.0
            ).values
        if m["style_match"].max() == 0:
            m["style_match"] = 1.0
    elif selected_clusters:
        m["style_match"] = df["cluster_name"].isin(selected_clusters).astype(float)
    else:
        # Không chọn phong cách cụ thể → mọi điểm đến trung lập như nhau
        m["style_match"] = 1.0

    # ── cluster_fit: độ gần tới tâm cụm K-Means (chỉ khi có query) ──
    aff = cluster_affinity(df, keyword_query, tfidf_model, tfidf_matrix)
    m["cluster_fit"] = aff if aff is not None else 1.0

    # ── community_score: mức chấp nhận học từ phản hồi người dùng trước ──
    if community and "destination_id" in df.columns:
        m["community_score"] = df["destination_id"].map(community).fillna(0.5).astype(float)
    else:
        m["community_score"] = 0.5

    return m


def rank_by_suitability(
    df: pd.DataFrame,
    budget=None,
    preferred_duration=None,
    selected_clusters=None,
    weights=None,
    keyword_query: str = None,
    tfidf_model=None,
    tfidf_matrix=None,
    community: dict = None,
    **kwargs
) -> pd.DataFrame:
    """
    Hàm chính: nhận DataFrame điểm đến (đã lọc sơ bộ), trả về DataFrame có thêm
    cột 'suitability_score' (0-100, % phù hợp) và đã sắp xếp giảm dần theo đó.
    Tự động hỗ trợ các bí danh tham số: user_budget, user_duration, user_style, vectorizer.
    """
    if budget is None:
        budget = kwargs.get("user_budget", kwargs.get("user_budget_input"))
    if preferred_duration is None:
        duration_val = kwargs.get("user_duration")
        if duration_val is not None and not isinstance(duration_val, (tuple, list)):
            preferred_duration = (max(0.5, duration_val - 1), duration_val + 1)
        elif duration_val is not None:
            preferred_duration = duration_val
    if keyword_query is None:
        keyword_query = kwargs.get("user_style", kwargs.get("style"))
    if tfidf_model is None:
        tfidf_model = kwargs.get("vectorizer")

    if df.empty:
        df = df.copy()
        df["suitability_score"] = pd.Series(dtype=float)
        return df

    criteria = {k: dict(v) for k, v in DEFAULT_CRITERIA.items()}
    if weights:
        for k, w in weights.items():
            # Tương thích ngược: "cost_avg" → "budget_fit"
            actual_key = "budget_fit" if k == "cost_avg" else k
            if actual_key in criteria:
                criteria[actual_key]["weight"] = w

    matrix = build_decision_matrix(
        df,
        preferred_duration=preferred_duration,
        selected_clusters=selected_clusters,
        budget=budget,
        keyword_query=keyword_query,
        tfidf_model=tfidf_model,
        tfidf_matrix=tfidf_matrix,
        community=community,
    )
    scores = topsis_score(matrix, criteria)

    out = df.copy()
    out["style_match"] = matrix["style_match"]
    out["cluster_fit"] = matrix["cluster_fit"]
    out["community_score"] = matrix["community_score"]
    for c in matrix.columns:  # đặc trưng từng tiêu chí, dùng để huấn luyện từ phản hồi
        out[f"f_{c}"] = matrix[c]
    out["topsis_raw"] = (scores * 100).round(2)
    out["suitability_score"] = (scores * 100).round(1)

    # Nếu người dùng nhập từ khóa: giữ điểm đến khớp trực tiếp (TF-IDF) HOẶC thuộc cụm
    # K-Means gần với sở thích đó. Trước đây chỉ lọc theo TF-IDF nên có truy vấn chỉ
    # còn 3 kết quả cho cả 3 miền.
    if keyword_query and tfidf_model is not None and tfidf_matrix is not None:
        keep = out["style_match"] >= 0.05
        if (matrix["cluster_fit"] < 1.0).any() or matrix["cluster_fit"].nunique() > 1:
            keep = keep | (matrix["cluster_fit"].reindex(out.index) >= CLUSTER_KEEP_THRESHOLD)
        out = out[keep]

    # Rescale điểm số của các địa điểm phù hợp (còn lại) lên khoảng 60% - 98%
    # để phù hợp với tâm lý người dùng (>= 50% là phù hợp)
    if len(out) > 0:
        max_s = out["suitability_score"].max()
        min_s = out["suitability_score"].min()
        if max_s > min_s:
            out["suitability_score"] = 60 + (out["suitability_score"] - min_s) / (max_s - min_s) * (98 - 60)
        else:
            out["suitability_score"] = 95.0
            
        out["suitability_score"] = out["suitability_score"].round(1)

    out = out.sort_values("suitability_score", ascending=False)
    return out


def explain_top_criteria(row: pd.Series, budget=None, keyword_query=None, feedback=None) -> list:
    """
    Sinh 2-3 lý do ngắn giải thích vì sao điểm đến phù hợp, dùng ở bước
    "Trợ giúp người dùng lựa chọn". Lý do được xếp theo mức liên quan tới yêu cầu.

    feedback: tuple (số lượt "phù hợp", số lượt "chưa phù hợp") của điểm đến này
              trong phản hồi người dùng trước (nếu có).
    """
    reasons = []

    sm = row.get("style_match")
    if keyword_query and pd.notna(sm) and sm >= 0.15:
        reasons.append(f"Khớp với yêu cầu bạn nhập (độ tương đồng nội dung {sm * 100:.0f}%)")
    elif keyword_query and pd.notna(row.get("cluster_fit")) and row.get("cluster_fit") >= CLUSTER_KEEP_THRESHOLD \
            and pd.notna(row.get("cluster_name")):
        reasons.append(f"Thuộc nhóm điểm đến gần với sở thích của bạn: {str(row.get('cluster_name')).strip()}")

    cost_min, cost_max = row.get("cost_min"), row.get("cost_max")
    if budget and pd.notna(cost_max) and cost_max <= budget:
        reasons.append("Phù hợp ngân sách của bạn")

    rating = row.get("rating_num")
    if pd.notna(rating) and rating >= 4.0:
        reasons.append(f"Được đánh giá cao (★ {rating:.1f}/5)")

    if feedback:
        fit, unfit = feedback
        if fit + unfit >= 2 and fit / (fit + unfit) >= 0.7:
            reasons.append(f"{fit}/{fit + unfit} người dùng trước đánh giá địa điểm này phù hợp")

    duration = row.get("duration_days")
    if pd.notna(duration):
        dur_val = float(duration)
        dur_str = f"{int(dur_val)}" if dur_val.is_integer() else f"{dur_val:.1f}"
        reasons.append(f"Thời gian tham quan lý tưởng (~{dur_str} ngày)")

    score = row.get("suitability_score")
    if pd.notna(score) and not reasons:
        reasons.append(f"Độ phù hợp tổng thể: {score:.0f}%")

    return reasons[:3]
