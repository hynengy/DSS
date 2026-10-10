import os
import re
import pandas as pd

def find_file(filename_candidates):
    """Tìm đường dẫn file hợp lệ từ danh sách các tên file/đường dẫn khả thi."""
    for path in filename_candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        f"Không tìm thấy file dữ liệu nào trong số: {filename_candidates}"
    )

def parse_rating(value):
    """'4.6/5' -> 4.6"""
    if pd.isna(value):
        return None
    match = re.search(r"[\d.]+", str(value))
    return float(match.group()) if match else None


USD_TO_VND = 25_000
_MONEY_UNIT = re.compile(r"^\s*(triệu|tr\b|nghìn|ngàn|k\b|usd|\$)")
_MONEY_TOKEN = re.compile(r"\d[\d.,]*")


def _money_unit(tail):
    m = _MONEY_UNIT.match(tail)
    if not m:
        return None
    u = m.group(1)
    if u in ("triệu", "tr"):
        return 1_000_000
    if u in ("nghìn", "ngàn", "k"):
        return 1_000
    return USD_TO_VND


def parse_money(text):
    """
    Chuỗi giá -> (min, max) VND. Đơn vị được xác định cho TỪNG số (vd "500k - 1.5 triệu"
    -> 500.000 và 1.500.000). Số không có đơn vị trong một khoảng thì mượn đơn vị của số
    phía sau ("600 - 1 triệu" -> 600k; "1.2 - 2.5 triệu" -> 1.2 triệu).
    "Miễn phí" chỉ áp dụng khi xuất hiện TRƯỚC con số đầu tiên (vé vào cổng miễn phí, còn
    các khoản phía sau là dịch vụ tùy chọn). Bỏ các số < 1.000đ (vd "1,3m", "4 cồn").
    """
    if pd.isna(text):
        return (0, 0)
    text = str(text).lower()
    first_num = _MONEY_TOKEN.search(text)
    free_pos = text.find("miễn phí")
    if free_pos >= 0 and (first_num is None or free_pos < first_num.start()):
        return (0, 0)

    toks = []  # (giá trị thô, đơn vị riêng hoặc None, có dấu phân cách nghìn)
    for m in _MONEY_TOKEN.finditer(text):
        tok = m.group().strip(".,")
        if not tok:
            continue
        grouped = bool(re.fullmatch(r"\d{1,3}([.,]\d{3})+", tok))
        val = float(tok.replace(".", "").replace(",", "")) if grouped else float(tok.replace(",", "."))
        toks.append([val, _money_unit(text[m.end():]), grouped])

    values = []
    for i, (val, unit, grouped) in enumerate(toks):
        if unit is None and not grouped and val < 1000:
            nxt = next((t[1] for t in toks[i + 1:] if t[1] is not None), None)
            if nxt == 1_000_000 and val >= 100:
                unit = 1_000          # "600 - 1 triệu" -> 600 nghìn
            else:
                unit = nxt
        v = val * (unit or 1)
        if v >= 1000:
            values.append(v)

    if not values:
        return (None, None)
    pair = values[:2]
    return (min(pair), max(pair))


_DUR_UNIT = re.compile(r"(phút|tiếng|giờ|ngày|buổi)")
HOURS_PER_DAY = 8.0  # một ngày tham quan ~ 8 tiếng


def parse_duration(text):
    """
    Thời lượng tham quan lý tưởng -> số NGÀY (1 ngày tham quan = 8 tiếng, 1 buổi = 0.5 ngày).
    Mỗi số lấy đơn vị đứng sau gần nhất: "45 phút – 1,5 tiếng", "30 - 45 phút".
    Bỏ nội dung trong ngoặc (giờ mở cửa "21:00", "5:30 - 8:00"...) và cụm "x đêm".
    """
    if pd.isna(text):
        return None
    t = str(text).lower()
    t = re.sub(r"\([^)]*\)", " ", t)
    t = re.sub(r"\d+\s*đêm", " ", t)
    t = t.replace("nửa ngày", "0.5 ngày").replace("cả ngày", "1 ngày")

    unit_days = {"phút": 1 / 60 / HOURS_PER_DAY, "tiếng": 1 / HOURS_PER_DAY,
                 "giờ": 1 / HOURS_PER_DAY, "ngày": 1.0, "buổi": 0.5}
    vals = []
    for m in re.finditer(r"\d+(?:[.,]\d+)?", t):
        u = _DUR_UNIT.search(t, m.end())
        if not u:
            continue
        vals.append(float(m.group().replace(",", ".")) * unit_days[u.group(1)])
    if vals:
        return round(sum(vals[:2]) / len(vals[:2]), 3)
    if re.search(r"buổi|sáng|chiều|tối", t):
        return 0.5
    return None


def parse_keywords(text):
    """'"UNESCO", "đảo", "hang động"' -> ['UNESCO', 'đảo', 'hang động']"""
    if pd.isna(text):
        return []
    return [k.strip().strip('"') for k in str(text).split(",")]


COORDS_LOOKUP = {
    "DEST_MN_58": (11.9036, 108.4489),
    "DEST_MN_59": (11.8549, 108.5494),
    "DEST_MN_60": (12.0465, 108.4383),
    "DEST_MN_61": (11.8887, 108.4326),
    "DEST_MN_62": (11.8906, 108.4172),
    "DEST_MN_63": (11.9404, 108.4452),
    "DEST_MN_64": (11.9419, 108.4552),
    "DEST_MN_65": (10.8413, 108.0673),
    "DEST_MN_66": (10.8450, 108.0620),
    "DEST_MN_67": (10.8520, 108.0700),
    "DEST_MN_68": (10.8580, 108.0750),
    "DEST_MN_69": (10.7798, 106.6990),
    "DEST_MN_70": (10.7799, 106.6999),
    "DEST_MN_71": (10.7770, 106.6953),
    "DEST_MN_72": (10.7767, 106.7032),
    "DEST_MN_73": (10.7682, 106.7068),
    "DEST_MN_74": (10.7738, 106.7036),
    "DEST_MN_75": (10.7725, 106.6980),
    "DEST_MN_76": (11.3789, 106.1683),
    "DEST_MN_77": (11.3039, 106.1347),
    "DEST_MN_78": (10.3582, 107.0701),
    "DEST_MN_79": (10.4572, 107.4412),
    "DEST_MN_80": (10.4902, 107.4876),
    "DEST_MN_81": (10.3265, 107.0844),
    "DEST_MN_82": (10.3235, 107.0818),
    "DEST_MN_83": (10.3496, 107.0691),
    "DEST_MN_84": (10.3262, 106.3533),
    "DEST_MN_85": (10.3082, 106.3351),
    "DEST_MN_86": (10.0055, 105.7485),
    "DEST_MN_87": (10.0336, 105.7876),
    "DEST_MN_88": (10.0682, 105.7484),
    "DEST_MN_89": (8.6835, 106.6080),
    "DEST_MN_90": (9.1768, 105.1504),
    "DEST_MN_91": (9.2392, 104.9723),
    "DEST_MN_92": (8.6086, 104.7176),
    "DEST_MN_93": (10.0125, 105.0809),
    "DEST_MN_94": (10.3833, 104.4833),
    "DEST_MN_95": (10.2899, 103.9840),
    "DEST_MN_96": (9.6806, 104.3542),
    "DEST_MN_97": (10.0157, 104.0175),
    "DEST_MN_98": (10.0289, 104.0094),
    "DEST_MN_99": (10.0545, 104.0322),
    "DEST_MN_100": (10.3347, 103.8569),
}


def clean_destinations(path=None):
    if path is None:
        path = find_file(
            [
                "Dataset - Danh sách Điểm tham quan (destinations).csv",
                "data/Dataset - Danh sách Điểm tham quan (destinations).csv",
                "destinations.csv",
                "data/destinations.csv",
            ]
        )

    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]

    df["rating_num"] = df["rating"].apply(parse_rating)
    df[["cost_min", "cost_max"]] = df["estimated_cost"].apply(
        lambda x: pd.Series(parse_money(x))
    )
    df["duration_days"] = df["ideal_duration"].apply(parse_duration)
    df["keywords_list"] = df["keywords"].apply(parse_keywords)

    # Điền tọa độ còn thiếu từ bảng tra cứu
    for idx, r in df.iterrows():
        did = r.get("destination_id")
        if did in COORDS_LOOKUP and (pd.isna(r.get("latitude")) or pd.isna(r.get("longitude"))):
            df.at[idx, "latitude"] = COORDS_LOOKUP[did][0]
            df.at[idx, "longitude"] = COORDS_LOOKUP[did][1]

    return df



def clean_hotels(path=None):
    if path is None:
        path = find_file(
            [
                "Dataset - Danh sách Khách sạn _ Nơi lưu trú (hotels).csv",
                "data/Dataset - Danh sách Khách sạn _ Nơi lưu trú (hotels).csv",
                "hotels.csv",
                "data/hotels.csv",
            ]
        )

    df = pd.read_csv(path)
    df.columns = [c.split("\n")[-1].strip() for c in df.columns]

    df["destination_id"] = df["destination_id"].ffill()
    df["destination_name"] = df["destination_name"].ffill()

    df["rating_num"] = df["rating"].apply(parse_rating)
    df[["price_min", "price_max"]] = df["price_range"].apply(
        lambda x: pd.Series(parse_money(x))
    )

    return df

def clean_restaurants(path=None):
    if path is None:
        path = find_file(
            [
                "Dataset - Danh sách Quán ăn _ Ẩm thực (restaurants).csv",
                "data/Dataset - Danh sách Quán ăn _ Ẩm thực (restaurants).csv",
                "restaurants.csv",
                "data/restaurants.csv",
            ]
        )

    df = pd.read_csv(path)
    df.columns = [c.split("\n")[-1].strip() for c in df.columns]

    df["destination_id"] = df["destination_id"].ffill()
    df["destination_name"] = df["destination_name"].ffill()
    df["province"] = df["province / location"].ffill()

    df["rating_num"] = df["rating"].apply(parse_rating)
    df[["price_min", "price_max"]] = df["price_level"].apply(
        lambda x: pd.Series(parse_money(x))
    )

    return df


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)

    dest_df = clean_destinations()
    hotel_df = clean_hotels()
    rest_df = clean_restaurants()

    print("=== Destinations ===")
    print(
        dest_df[
            [
                "destination_name",
                "rating_num",
                "cost_min",
                "cost_max",
                "duration_days",
            ]
        ].head()
    )

    # Xuất dữ liệu đã làm sạch vào thư mục data/
    dest_df.to_csv(
        "data/clean_destinations.csv", index=False, encoding="utf-8-sig"
    )
    hotel_df.to_csv("data/clean_hotels.csv", index=False, encoding="utf-8-sig")
    rest_df.to_csv(
        "data/clean_restaurants.csv", index=False, encoding="utf-8-sig"
    )

    with pd.ExcelWriter(
        "data/Cleaned_Travel_Dataset.xlsx", engine="openpyxl"
    ) as writer:
        dest_df.to_excel(writer, sheet_name="Destinations", index=False)
        hotel_df.to_excel(writer, sheet_name="Hotels", index=False)
        rest_df.to_excel(writer, sheet_name="Restaurants", index=False)

    print("\n-> Đã xuất thành công Dữ liệu làm sạch vào thư mục 'data/'!")