import json
import os
from datetime import datetime

FEEDBACK_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "feedback_store.json")

DEFAULT_SEED_FEEDBACK = {
    "implicit_views": {},
    "explicit_reviews": []
}


def _ensure_feedback_file():
    if not os.path.exists(FEEDBACK_FILE):
        os.makedirs(os.path.dirname(FEEDBACK_FILE), exist_ok=True)
        with open(FEEDBACK_FILE, "w", encoding="utf-8") as f:
            json.dump(DEFAULT_SEED_FEEDBACK, f, ensure_ascii=False, indent=2)


def load_feedback_data():
    _ensure_feedback_file()
    try:
        with open(FEEDBACK_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            if "implicit_views" not in data:
                data["implicit_views"] = {}
            if "explicit_reviews" not in data:
                data["explicit_reviews"] = []
            return data
    except Exception:
        return DEFAULT_SEED_FEEDBACK


def save_feedback_data(data):
    _ensure_feedback_file()
    with open(FEEDBACK_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def log_implicit_view(dest_id: str, dest_name: str, region: str):
    data = load_feedback_data()
    views = data.get("implicit_views", {})
    if dest_id in views:
        views[dest_id]["views"] += 1
    else:
        views[dest_id] = {"name": dest_name, "region": region, "views": 1}
    data["implicit_views"] = views
    save_feedback_data(data)


def save_explicit_feedback(persona: str, stars: int, fit_items: list, unfit_items: list, comment: str,
                           items: list = None, context: dict = None):
    """
    items:   [{dest_id, name, rank, label ('fit'|'unfit'), features{tiêu chí TOPSIS}}]
             -> dữ liệu huấn luyện cho utils/learning.py và tính Precision@5 thật.
    context: bối cảnh khảo sát (ngân sách, số ngày, câu mô tả) tại thời điểm đánh giá.
    """
    data = load_feedback_data()
    reviews = data.get("explicit_reviews", [])
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    reviews.insert(0, {
        "timestamp": now_str,
        "persona": persona,
        "stars": stars,
        "fit_items": fit_items,
        "unfit_items": unfit_items,
        "comment": comment,
        **({"items": items} if items else {}),
        **({"context": context} if context else {}),
    })
    data["explicit_reviews"] = reviews
    save_feedback_data(data)


def get_feedback_stats():
    data = load_feedback_data()
    reviews = data.get("explicit_reviews", [])
    views_dict = data.get("implicit_views", {})

    total_explicit = len(reviews)
    avg_stars = round(sum(r["stars"] for r in reviews) / total_explicit, 1) if total_explicit > 0 else 0.0
    total_views = sum(v["views"] for v in views_dict.values())

    implicit_list = [
        {"dest_id": k, "name": v["name"], "region": v["region"], "views": v["views"]}
        for k, v in views_dict.items()
    ]
    implicit_list.sort(key=lambda x: x["views"], reverse=True)

    return {
        "total_explicit": total_explicit,
        "avg_stars": avg_stars,
        "total_views": total_views,
        "implicit_list": implicit_list,
        "explicit_reviews": reviews
    }
