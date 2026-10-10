"""
ĐÁNH GIÁ HỆ THỐNG (bước 7 của quy trình DSS)

Có 3 nhóm chỉ số:

A. Đánh giá THẬT từ người dùng (online): Precision@5, Hit-rate@5, điểm sao trung bình,
   tính trên phản hồi "Phù hợp / Chưa phù hợp" mà người dùng gắn cho top-5.

B. Đánh giá offline theo KỊCH BẢN, so sánh hệ thống với các baseline. Không có nhãn
   người dùng thật ở quy mô lớn nên "phù hợp" được định nghĩa bằng chính các ràng buộc
   khách nêu: TỔNG chi phí ước tính (vé vào cổng x số người + khách sạn, cùng công thức
   với TOPSIS) <= ngân sách cả nhóm, thời lượng lệch <= 1 ngày, và (nếu có mô tả)
   tên/từ khóa gán tay của điểm đến chứa từ khóa của ý định. Định nghĩa này không dùng
   mô tả văn bản nên độc lập với TF-IDF của bộ xếp hạng. Kết quả chỉ nói hệ thống đáp
   ứng yêu cầu đã nêu tốt đến đâu, KHÔNG thay cho đánh giá của người dùng thật.
   Độ tin cậy: khoảng tin cậy bootstrap 95% trên các kịch bản + kiểm định Wilcoxon cặp
   đôi (DSS so với baseline mạnh nhất), để chắc chắn khoảng cách không do ngẫu nhiên.

C. Chất lượng & độ ổn định phân cụm: silhouette, kích thước cụm, ARI giữa các lần chạy.
"""

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import silhouette_score

from utils.topsis import rank_by_suitability, recommended_weights, estimate_trip_cost

K = 5
N_BOOT = 1000
NUM_PEOPLE = 2  # kịch bản chuẩn: nhóm 2 người (1 phòng)

# ý định -> từ khóa để xác định "đúng chủ đề" (chỉ dò trong TÊN + TỪ KHÓA gán tay)
INTENT_TERMS = {
    "thiên nhiên xanh mát": ["thiên nhiên", "xanh", "núi", "rừng", "thác", "suối", "hồ", "vườn quốc gia", "sinh thái", "đồi"],
    "biển đảo": ["biển", "đảo", "vịnh", "bãi", "cát trắng", "san hô", "hòn"],
    "lịch sử văn hóa": ["lịch sử", "văn hóa", "di tích", "di sản", "cổ", "bảo tàng", "kiến trúc", "thành", "phố cổ"],
    "chùa tâm linh": ["chùa", "tâm linh", "đền", "miếu", "thiền", "lăng", "linh thiêng"],
    "nghỉ dưỡng thư giãn": ["nghỉ dưỡng", "thư giãn", "resort", "spa", "yên tĩnh", "bãi biển"],
    "núi thác": ["núi", "thác", "suối", "đèo", "cao nguyên", "đồi"],
}
BUDGETS = [1_000_000, 2_000_000, 3_000_000, 5_000_000, 8_000_000]
DURATIONS = [0.5, 1.0, 2.0, 3.0]


def _tag_text(df):
    kws = df["keywords_list"].apply(lambda k: " ".join(k) if isinstance(k, (list, tuple)) else str(k))
    return (df["destination_name"].fillna("") + " " + kws).str.lower()


def _relevance(df, tag_text, budget, duration, intent, hotel_df=None, num_people=NUM_PEOPLE):
    """
    Trả về (graded 0..1, binary 0/1, intent_ok, budget_ok, duration_ok) theo index df.
    budget = ngân sách / NGƯỜI; so với tổng chi phí cả nhóm (cùng công thức với TOPSIS).
    """
    pref = (max(0.5, duration - 1), duration + 1)
    cost = estimate_trip_cost(df, pref, hotel_df, num_people)
    b_ok = (cost <= budget * num_people).astype(float)
    d_ok = ((df["duration_days"].fillna(duration) - duration).abs() <= 1).astype(float)
    parts = [b_ok, d_ok]
    if intent:
        terms = INTENT_TERMS[intent]
        i_ok = tag_text.apply(lambda t: any(x in t for x in terms)).astype(float)
        parts.append(i_ok)
    else:
        i_ok = pd.Series(np.nan, index=df.index)
    graded = sum(parts) / len(parts)
    binary = (graded == 1.0).astype(float)
    return graded, binary, i_ok, b_ok, d_ok


def _ndcg(gains_ranked, all_gains, k=K):
    disc = 1.0 / np.log2(np.arange(2, k + 2))
    g = np.array(list(gains_ranked)[:k] + [0.0] * max(0, k - len(gains_ranked)))
    ideal = np.sort(np.array(all_gains))[::-1][:k]
    ideal = np.pad(ideal, (0, k - len(ideal)))
    idcg = (ideal * disc).sum()
    return float((g * disc).sum() / idcg) if idcg > 0 else np.nan


def _bootstrap_ci(values, n_boot=N_BOOT, seed=0):
    v = np.asarray([x for x in values if not np.isnan(x)], dtype=float)
    if len(v) == 0:
        return (np.nan, np.nan)
    rng = np.random.RandomState(seed)
    means = v[rng.randint(0, len(v), size=(n_boot, len(v)))].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


SYSTEM_NAME = "Hệ thống DSS (TOPSIS + K-Means + phản hồi)"
ORACLE_NAME = "Giới hạn trên (xếp lý tưởng)"


def offline_evaluation(df, tfidf_model, tfidf_matrix, community=None, seed=42, hotel_df=None,
                       num_people=NUM_PEOPLE):
    """
    df: DataFrame điểm đến đã có cluster_id/cluster_name (kết quả run_kmeans),
        thứ tự dòng khớp với tfidf_matrix.
    Trả về (bảng so sánh, số kịch bản, thống kê độ tin cậy).
    """
    from sklearn.metrics.pairwise import cosine_similarity
    from utils.topsis import expand_query_terms

    df = df.reset_index(drop=True)
    tag_text = _tag_text(df)
    rng = np.random.RandomState(seed)
    rating = df["rating_num"].fillna(df["rating_num"].median())

    scenarios = [(b, d, it) for b in BUDGETS for d in DURATIONS for it in list(INTENT_TERMS) + [None]]

    def rank_random(b, d, it):
        return list(rng.permutation(len(df)))

    def rank_rating(b, d, it):
        return list(np.argsort(-rating.values, kind="stable"))

    def rank_tfidf(b, d, it):
        if not it:
            return rank_rating(b, d, it)
        sims = cosine_similarity(tfidf_model.transform([expand_query_terms(it)]), tfidf_matrix).ravel()
        return list(np.argsort(-sims, kind="stable"))

    def rank_system(b, d, it):
        out = rank_by_suitability(
            df, weights=recommended_weights(bool(it)), budget=b * num_people, num_people=num_people,
            preferred_duration=(max(0.5, d - 1), d + 1), keyword_query=it or "",
            tfidf_model=tfidf_model, tfidf_matrix=tfidf_matrix, community=community, hotel_df=hotel_df,
        )
        return list(out.index)

    def rank_oracle(b, d, it, graded=None):
        return list(np.argsort(-graded.values, kind="stable"))

    systems = {
        "Ngẫu nhiên": rank_random,
        "Chỉ xếp theo rating": rank_rating,
        "Chỉ TF-IDF (content-based)": rank_tfidf,
        SYSTEM_NAME: rank_system,
        ORACLE_NAME: rank_oracle,
    }

    acc = {name: {"p": [], "ndcg": [], "intent": [], "budget": [], "dur": [], "div": [], "seen": set()}
           for name in systems}
    for (b, d, it) in scenarios:
        graded, binary, i_ok, b_ok, d_ok = _relevance(df, tag_text, b, d, it, hotel_df, num_people)
        for name, fn in systems.items():
            top = (fn(b, d, it, graded=graded) if fn is rank_oracle else fn(b, d, it))[:K]
            a = acc[name]
            a["p"].append(binary.iloc[top].sum() / K)
            a["ndcg"].append(_ndcg(graded.iloc[top].tolist(), graded.tolist()))
            if it:
                a["intent"].append(i_ok.iloc[top].sum() / K)
            a["budget"].append(b_ok.iloc[top].sum() / K)
            a["dur"].append(d_ok.iloc[top].sum() / K)
            a["div"].append(df["cluster_id"].iloc[top].nunique() / K)
            a["seen"].update(top)

    rows = []
    for name, a in acc.items():
        lo, hi = _bootstrap_ci(a["ndcg"])
        rows.append({
            "Hệ thống": name,
            f"Precision@{K}": round(float(np.nanmean(a["p"])), 3),
            f"NDCG@{K}": round(float(np.nanmean(a["ndcg"])), 3),
            f"NDCG@{K} KTC 95%": f"[{lo:.3f} – {hi:.3f}]",
            "Đúng chủ đề": round(float(np.mean(a["intent"])), 3),
            "Trong ngân sách": round(float(np.mean(a["budget"])), 3),
            "Đúng thời lượng": round(float(np.mean(a["dur"])), 3),
            "Đa dạng cụm": round(float(np.mean(a["div"])), 3),
            "Độ phủ danh mục": round(len(a["seen"]) / len(df), 3),
        })

    # Kiểm định Wilcoxon cặp đôi (cùng kịch bản) giữa DSS và baseline tốt nhất
    baselines = [n for n in systems if n not in (SYSTEM_NAME, ORACLE_NAME)]
    best_base = max(baselines, key=lambda n: np.nanmean(acc[n]["ndcg"]))
    sys_v, base_v = np.array(acc[SYSTEM_NAME]["ndcg"]), np.array(acc[best_base]["ndcg"])
    mask = ~(np.isnan(sys_v) | np.isnan(base_v))
    diff = sys_v[mask] - base_v[mask]
    try:
        p_value = float(wilcoxon(sys_v[mask], base_v[mask]).pvalue) if np.any(diff != 0) else 1.0
    except ValueError:
        p_value = 1.0
    d_lo, d_hi = _bootstrap_ci(diff)
    stats = {
        "baseline": best_base,
        "delta_ndcg": float(diff.mean()),
        "delta_ci": (d_lo, d_hi),
        "p_value": p_value,
        "win": int((diff > 0).sum()), "tie": int((diff == 0).sum()), "loss": int((diff < 0).sum()),
        "num_people": num_people,
    }
    return pd.DataFrame(rows), len(scenarios), stats


def online_metrics(reviews):
    """
    Chỉ số từ đánh giá thật của người dùng.
    Chỉ tính các đánh giá định dạng mới (có "items"), vì đánh giá cũ được thu khi nút
    "Phù hợp" và 5 sao là mặc định nên luôn ra Precision 100% (thiên lệch).
    """
    prec, hits, stars = [], [], []
    reviews = [rv for rv in reviews if rv.get("items")]
    for rv in reviews:
        n_fit, n_unfit = len(rv.get("fit_items", [])), len(rv.get("unfit_items", []))
        if n_fit + n_unfit > 0:
            prec.append(n_fit / (n_fit + n_unfit))
            hits.append(1.0 if n_fit > 0 else 0.0)
        if "stars" in rv:
            stars.append(rv["stars"])
    return {
        "n_reviews": len(reviews),
        "precision_at_5": float(np.mean(prec)) if prec else None,
        "hit_rate": float(np.mean(hits)) if hits else None,
        "avg_stars": float(np.mean(stars)) if stars else None,
    }


def clustering_report(X, labels, stability=None):
    sizes = pd.Series(labels).value_counts().sort_index()
    rep = {
        "silhouette": float(silhouette_score(X, labels)),
        "k": int(len(sizes)),
        "sizes": sizes.tolist(),
        "min_size": int(sizes.min()),
    }
    if stability:
        rep.update(stability)
    return rep
