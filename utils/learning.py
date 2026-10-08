"""
HỌC TỪ HÀNH VI NGƯỜI DÙNG (bước 2-3 của quy trình DSS)

Hệ thống thu phản hồi tường minh (mỗi lần gửi đánh giá: top-5 địa điểm được gắn nhãn
"Phù hợp"/"Chưa phù hợp") và dùng nó theo 2 cách:

1. community_scores(): mức chấp nhận của từng điểm đến, làm mượt Beta(1,1)
       score = (fit + 1) / (fit + unfit + 2)      -> chưa có phản hồi = 0.5 (trung lập)
   được đưa vào TOPSIS như tiêu chí "community_score".

2. learn_weights(): huấn luyện Logistic Regression dự đoán P(phù hợp) từ các tiêu chí
   TOPSIS (rating, budget_fit, duration_fit, style_match, cluster_fit) đã ghi lại lúc
   người dùng đánh giá. Hệ số dương được chuyển thành trọng số và trộn với trọng số
   khởi điểm theo lambda = n / (n + 100): ít dữ liệu -> gần như giữ trọng số mặc định,
   nhiều dữ liệu -> trọng số do dữ liệu quyết định. Dưới MIN_LABELS nhãn thì KHÔNG học
   (tránh học trên dữ liệu quá ít).
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

LEARN_FEATURES = ["rating_num", "budget_fit", "duration_fit", "style_match", "cluster_fit"]
MIN_LABELS = 30
LAMBDA_HALF = 100  # lambda = n / (n + LAMBDA_HALF)


def item_key(name, region):
    return f"{str(name).strip()} ({region})"


def build_name_index(dest_df):
    """'Tên (Miền)' -> destination_id, để đọc các đánh giá cũ chỉ lưu tên."""
    return {item_key(r["destination_name"], r["Miền"]): r["destination_id"] for _, r in dest_df.iterrows()}


def _iter_labeled(reviews, name_index=None, include_legacy=False):
    """
    Sinh (destination_id, label 1/0) từ các đánh giá.
    Chỉ dùng định dạng mới (có "items": người dùng chọn nhãn rõ ràng). Đánh giá cũ (chỉ có tên
    địa điểm) được tạo khi nút mặc định là "Phù hợp" nên nhãn thiên lệch -> mặc định bỏ qua.
    """
    name_index = name_index or {}
    for rv in reviews:
        if rv.get("items"):
            for it in rv["items"]:
                yield it.get("dest_id"), 1 if it.get("label") == "fit" else 0
        elif include_legacy:
            for nm in rv.get("fit_items", []):
                yield name_index.get(nm), 1
            for nm in rv.get("unfit_items", []):
                yield name_index.get(nm), 0


def community_scores(reviews, name_index=None, a=1.0, b=1.0, include_legacy=False):
    """Trả về (scores {dest_id: 0..1}, counts {dest_id: (fit, unfit)})."""
    fit, unfit = {}, {}
    for dest_id, label in _iter_labeled(reviews, name_index, include_legacy):
        if not dest_id:
            continue
        (fit if label else unfit)[dest_id] = (fit if label else unfit).get(dest_id, 0) + 1
    ids = set(fit) | set(unfit)
    scores = {d: (fit.get(d, 0) + a) / (fit.get(d, 0) + unfit.get(d, 0) + a + b) for d in ids}
    counts = {d: (fit.get(d, 0), unfit.get(d, 0)) for d in ids}
    return scores, counts


def collect_training_data(reviews):
    rows, y = [], []
    for rv in reviews:
        for it in rv.get("items", []) or []:
            f = it.get("features")
            if f and all(k in f for k in LEARN_FEATURES) and it.get("label") in ("fit", "unfit"):
                rows.append([float(f[k]) for k in LEARN_FEATURES])
                y.append(1 if it["label"] == "fit" else 0)
    return np.array(rows), np.array(y)


def learn_weights(reviews, base_weights, min_labels=MIN_LABELS):
    """
    Trả về (weights, info). info: {status, n_labels, n_fit, n_unfit, lambda, cv_auc, coefs}.
    status: 'learned' | 'insufficient_data' | 'single_class'
    """
    X, y = collect_training_data(reviews)
    info = {"status": "insufficient_data", "n_labels": int(len(y)), "n_fit": int(y.sum()) if len(y) else 0,
            "n_unfit": int(len(y) - y.sum()) if len(y) else 0, "lambda": 0.0, "cv_auc": None,
            "coefs": {}, "min_labels": min_labels}
    weights = dict(base_weights)
    if len(y) < min_labels:
        return weights, info
    if len(set(y.tolist())) < 2:
        info["status"] = "single_class"
        return weights, info

    model = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000))
    model.fit(X, y)
    coef = model[-1].coef_[0]
    info["coefs"] = {k: float(c) for k, c in zip(LEARN_FEATURES, coef)}

    min_class = int(min(y.sum(), len(y) - y.sum()))
    if min_class >= 5:
        try:
            cv = StratifiedKFold(n_splits=min(5, min_class), shuffle=True, random_state=42)
            info["cv_auc"] = float(cross_val_score(model, X, y, cv=cv, scoring="roc_auc").mean())
        except Exception:
            info["cv_auc"] = None

    pos = np.clip(coef, 0, None)
    if pos.sum() <= 0:
        info["status"] = "single_class"  # không có tiêu chí nào dự báo được -> giữ mặc định
        return weights, info

    budget_mass = sum(base_weights.get(k, 0.0) for k in LEARN_FEATURES)
    learned = {k: float(p / pos.sum() * budget_mass) for k, p in zip(LEARN_FEATURES, pos)}
    lam = len(y) / (len(y) + LAMBDA_HALF)
    for k in LEARN_FEATURES:
        weights[k] = lam * learned[k] + (1 - lam) * base_weights.get(k, 0.0)
    info.update(status="learned", **{"lambda": float(lam)})
    return weights, info
