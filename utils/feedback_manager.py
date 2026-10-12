import json
import os
import time
from datetime import datetime
import requests

try:
    import streamlit as st
except ImportError:
    st = None

FEEDBACK_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "feedback_store.json")

DEFAULT_SEED_FEEDBACK = {
    "implicit_views": {},
    "explicit_reviews": []
}

_MEMORY_CACHE = {"data": None, "timestamp": 0}
CACHE_TTL_SECONDS = 10


def get_webhook_url() -> str:
    """Lấy URL Webhook kết nối Google Sheets (từ secrets.toml hoặc biến môi trường)."""
    if st is not None:
        try:
            if "FEEDBACK_WEBHOOK_URL" in st.secrets:
                return str(st.secrets["FEEDBACK_WEBHOOK_URL"]).strip()
        except Exception:
            pass
    return os.environ.get("FEEDBACK_WEBHOOK_URL", "").strip()


def is_cloud_configured() -> bool:
    """Kiểm tra xem hệ thống đã được cấu hình URL Google Sheets hay chưa."""
    return bool(get_webhook_url())


def _ensure_feedback_file():
    if not os.path.exists(FEEDBACK_FILE):
        os.makedirs(os.path.dirname(FEEDBACK_FILE), exist_ok=True)
        with open(FEEDBACK_FILE, "w", encoding="utf-8") as f:
            json.dump(DEFAULT_SEED_FEEDBACK, f, ensure_ascii=False, indent=2)


def _load_local_feedback():
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
        return {
            "implicit_views": {},
            "explicit_reviews": []
        }


def _save_local_feedback(data):
    _ensure_feedback_file()
    try:
        with open(FEEDBACK_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def load_feedback_data(force_refresh=False):
    """
    Đọc dữ liệu đánh giá và lượt xem:
    Ưu tiên đọc từ Google Sheets (nếu có cấu hình webhook) để đồng bộ trên cloud.
    Tự động fallback về file JSON cục bộ nếu chưa cấu hình hoặc mất mạng.
    """
    webhook_url = get_webhook_url()
    now = time.time()

    # Sử dụng cache bộ nhớ ngắn hạn để tránh gọi HTTP liên tục trong cùng 1 trang
    if not force_refresh and _MEMORY_CACHE["data"] is not None and (now - _MEMORY_CACHE["timestamp"] < CACHE_TTL_SECONDS):
        return _MEMORY_CACHE["data"]

    if webhook_url:
        try:
            resp = requests.get(webhook_url, timeout=4)
            if resp.status_code == 200:
                cloud_data = resp.json()
                if isinstance(cloud_data, dict) and "explicit_reviews" in cloud_data:
                    # Đồng bộ ngược vào file local để có bản lưu offline
                    _save_local_feedback(cloud_data)
                    _MEMORY_CACHE["data"] = cloud_data
                    _MEMORY_CACHE["timestamp"] = now
                    return cloud_data
        except Exception:
            # Gặp lỗi kết nối cloud thì fallback về file local
            pass

    local_data = _load_local_feedback()
    _MEMORY_CACHE["data"] = local_data
    _MEMORY_CACHE["timestamp"] = now
    return local_data


def save_feedback_data(data):
    _MEMORY_CACHE["data"] = data
    _MEMORY_CACHE["timestamp"] = time.time()
    _save_local_feedback(data)


def log_implicit_view(dest_id: str, dest_name: str, region: str):
    data = load_feedback_data()
    views = data.get("implicit_views", {})
    if dest_id in views:
        views[dest_id]["views"] += 1
    else:
        views[dest_id] = {"name": dest_name, "region": region, "views": 1}
    data["implicit_views"] = views
    save_feedback_data(data)

    # Gửi ngầm lên cloud nếu có cấu hình
    webhook_url = get_webhook_url()
    if webhook_url:
        try:
            requests.post(
                webhook_url,
                json={
                    "action": "log_view",
                    "view": {"dest_id": dest_id, "name": dest_name, "region": region}
                },
                timeout=3
            )
        except Exception:
            pass


def save_explicit_feedback(persona: str, stars: int, fit_items: list, unfit_items: list, comment: str,
                           items: list = None, context: dict = None):
    """
    Lưu đánh giá tường minh từ người dùng.
    Đồng thời lưu cục bộ và gửi lên Google Sheets (nếu đã kết nối).
    """
    data = load_feedback_data(force_refresh=True)
    reviews = data.get("explicit_reviews", [])
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    new_review = {
        "timestamp": now_str,
        "persona": persona,
        "stars": stars,
        "fit_items": fit_items,
        "unfit_items": unfit_items,
        "comment": comment,
        **({"items": items} if items else {}),
        **({"context": context} if context else {}),
    }
    reviews.insert(0, new_review)
    data["explicit_reviews"] = reviews
    save_feedback_data(data)

    # Đẩy lên Google Sheets
    webhook_url = get_webhook_url()
    if webhook_url:
        try:
            requests.post(
                webhook_url,
                json={"action": "save_explicit", "review": new_review},
                timeout=4
            )
        except Exception:
            pass


def sync_local_to_cloud():
    """Đẩy toàn bộ dữ liệu hiện tại lên Google Sheets."""
    webhook_url = get_webhook_url()
    if not webhook_url:
        return False, "Chưa cấu hình URL kết nối Google Sheets."

    local_data = _load_local_feedback()
    try:
        resp = requests.post(
            webhook_url,
            json={
                "action": "sync_all",
                "explicit_reviews": local_data.get("explicit_reviews", []),
                "implicit_views": local_data.get("implicit_views", {})
            },
            timeout=8
        )
        if resp.status_code == 200:
            _MEMORY_CACHE["data"] = None  # Xóa cache để lần sau nạp mới
            return True, "Đã đồng bộ toàn bộ dữ liệu lên Google Sheets thành công!"
        return False, f"Máy chủ Google Sheets phản hồi mã: {resp.status_code}"
    except Exception as e:
        return False, f"Lỗi kết nối: {str(e)}"


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


def reset_feedback_data():
    """Xóa toàn bộ dữ liệu đánh giá và lượt xem về trạng thái trống ban đầu."""
    empty_data = {
        "implicit_views": {},
        "explicit_reviews": []
    }
    save_feedback_data(empty_data)

    webhook_url = get_webhook_url()
    if webhook_url:
        try:
            requests.post(webhook_url, json={"action": "reset"}, timeout=4)
        except Exception:
            pass

    return empty_data
