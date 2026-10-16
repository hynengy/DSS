# HỆ THỐNG TRỢ GIÚP RA QUYẾT ĐỊNH DU LỊCH VIỆT NAM (VIETNAM TRAVEL DSS)

> **Báo cáo đề tài môn học: Hệ trợ giúp ra quyết định (Decision Support Systems - DSS)**  
> 🌐 **Link chạy thử nghiệm trực tuyến (Live Demo):** [https://travel-dss.streamlit.app/](https://travel-dss.streamlit.app/)  
> 💻 **Công nghệ cốt lõi:** Python 3.9+, Streamlit, Scikit-learn, SciPy, PyDeck 3D, Google Sheets API / Webhook.

---

## 👥 BẢNG PHÂN CÔNG CÔNG VIỆC NHÓM

| STT | Họ và tên | Mã sinh viên | Nội dung công việc đảm nhiệm | Tỉ lệ đóng góp (%) |
| :---: | :--- | :---: | :--- | :---: |
| **1** | **Nguyễn Thúy Hiền** | **2355010073** | • Thu thập và tiền xử lý dữ liệu (Dataset).<br>• Xây dựng và triển khai thuật toán phân cụm K-Means, tính độ tương đồng Cosine Similarity và phương pháp đánh giá đa tiêu chí TOPSIS.<br>• Lập trình mô hình Logistic Regression và cơ chế học thích nghi.<br>• Xây dựng giao diện Streamlit.<br>• Viết báo cáo Word và làm slide PowerPoint. | **25%** |
| **2** | **Lê Thị Dịu** | **2355010037** | • Thu thập và tiền xử lý dữ liệu (Dataset).<br>• Xây dựng ma trận quyết định và thuật toán chuẩn hóa vector.<br>• Xây dựng tính năng giải trình kết quả và dự toán chi phí trọn gói.<br>• Phối hợp xây dựng kịch bản kiểm thử mô hình.<br>• Hỗ trợ làm báo cáo Word và làm slide PowerPoint. | **21%** |
| **3** | **Lê Ngọc Duy** | **2355010041** | • Tham gia thu thập dữ liệu (Dataset).<br>• Xây dựng và tối ưu hóa phương pháp biểu diễn văn bản TF-IDF.<br>• Xử lý dữ liệu định lượng và các ràng buộc về ngân sách, thời lượng. | **18%** |
| **4** | **Lê Tiến Quân** | **2355010161** | • Tham gia thu thập dữ liệu (Dataset).<br>• Triển khai cơ chế làm mượt điểm tín nhiệm cộng đồng bằng phân phối Beta.<br>• Thực hiện kiểm thử các chức năng hệ thống. | **18%** |
| **5** | **Nguyễn Việt Anh** | **2355010013** | • Tham gia thu thập dữ liệu (Dataset).<br>• Xây dựng các chỉ số đánh giá xếp hạng và so sánh với baseline.<br>• Thiết kế kiến trúc tổng thể của hệ thống. | **18%** |
| **TỔNG** | | | **Toàn bộ 5 thành viên hoàn thành đầy đủ nhiệm vụ được giao** | **100%** |

---

## 🌟 ĐƯỜNG DẪN TRẢI NGHIỆM HỆ THỐNG
* **Ứng dụng đã triển khai (Cloud Deployment):**  
  👉 **[https://travel-dss.streamlit.app/](https://travel-dss.streamlit.app/)**  
  *(Hệ thống đã được cấu hình Cloud Webhook kết nối Google Sheets tự động, người dùng có thể trải nghiệm toàn bộ tính năng gợi ý, tìm kiếm ngữ nghĩa, giải trình quyết định, đánh giá và học thích nghi mà không cần cài đặt môi trường).*

---

## 📖 1. TỔNG QUAN ĐỀ TÀI & BÀI TOÁN RA QUYẾT ĐỊNH

### 1.1. Bối cảnh & Thách thức
Du lịch Việt Nam ngày càng phát triển với hàng trăm điểm đến phong phú trải dài từ Bắc chí Nam. Tuy nhiên, du khách đối mặt với bài toán ra quyết định phức tạp:
1. **Quá tải thông tin:** Khó khăn khi tổng hợp và so sánh đặc điểm điểm đến, khách sạn phù hợp và nhà hàng lân cận.
2. **Nhiều tiêu chí mâu thuẫn (MCDM):** Cần cân bằng đồng thời giữa **ngân sách tối đa**, **thời gian chuyến đi**, **sở thích phong cách** (biển đảo, sinh thái, di tích, tâm linh, nghỉ dưỡng...) và **chất lượng trải nghiệm** (rating, uy tín thực tế).
3. **Tính thích nghi của hệ thống:** Các hệ thống gợi ý truyền thống thường cố định trọng số, không tự học từ phản hồi khen/chê của du khách sau khi trải nghiệm.

### 1.2. Mục tiêu của hệ thống DSS
Hệ thống **Vietnam Travel DSS** cung cấp giải pháp trợ giúp ra quyết định toàn diện:
* **Gợi ý điểm đến tối ưu:** Phân cụm đặc trưng chủ đề (**K-Means + TF-IDF**) và xếp hạng đa tiêu chí (**TOPSIS**).
* **Đề xuất Combo trọn gói (Điểm đến + Khách sạn + Quán ăn):** Tự động ghép nối các dịch vụ tương ứng tại địa phương, dự toán chi phí thực tế cho cả nhóm du khách.
* **Minh bạch hóa quyết định (Explainable DSS):** Giải trình rõ ràng lý do điểm đến được xếp hạng cao dựa trên các tiêu chí nổi bật.
* **Học thích nghi tự động (Adaptive Learning):** Ứng dụng **Logistic Regression** và làm mượt **Beta Bayes** để tự động điều chỉnh trọng số tiêu chí theo phản hồi của cộng đồng.
* **Đánh giá khoa học:** So sánh với 3 mô hình đối chuẩn (Baseline), kiểm định giả thuyết thống kê **Wilcoxon** và khoảng tin cậy **Bootstrap 95%**.

---

## 📂 2. CẤU TRÚC THƯ MỤC DỰ ÁN

Dự án được chuẩn hóa theo mô hình kiến trúc phân tầng rõ ràng:

```text
DSS/
│
├── app.py                      # Ứng dụng Web chính (Giao diện Streamlit tương tác)
├── run.bat                     # Script 1-click khởi chạy nhanh trên Windows
├── requirements.txt            # Danh sách các thư viện phụ thuộc (Dependencies)
├── README.md                   # Báo cáo chi tiết đề tài & tài liệu hướng dẫn
├── google_sheets_script.js     # Mã nguồn Google Apps Script backend đồng bộ Google Sheets
│
├── .streamlit/                 # Cấu hình giao diện và môi trường Streamlit
│   ├── config.toml             # Theme giao diện cổ điển (Font Serif, Nâu đỏ & Kem)
│   ├── secrets.toml            # Biến môi trường webhook Google Sheets
│   └── secrets.toml.example    # Mẫu cấu hình webhook tham khảo
│
├── data/                       # Dữ liệu nguồn và cơ sở tri thức của hệ thống
│   ├── destinations.csv        # 100 điểm đến du lịch Việt Nam (Dữ liệu gốc)
│   ├── hotels.csv              # 180+ khách sạn & nơi lưu trú (Dữ liệu gốc)
│   ├── restaurants.csv         # 200+ quán ăn & nhà hàng đặc sản (Dữ liệu gốc)
│   ├── clean_destinations.csv  # Điểm đến đã làm sạch và điền tọa độ GPS
│   ├── clean_hotels.csv        # Khách sạn đã làm sạch giá và tên chuẩn hóa
│   ├── clean_restaurants.csv   # Quán ăn đã làm sạch và phân loại mức giá
│   ├── Cleaned_Travel_Dataset.xlsx # File Excel 3 sheet tổng hợp dữ liệu chuẩn hóa
│   └── feedback_store.json     # Cơ sở dữ liệu lưu trữ lượt xem và đánh giá của người dùng
│
└── utils/                      # Các module thuật toán cốt lõi của DSS
    ├── __init__.py             # Khởi tạo gói module utils
    ├── preprocessing.py        # Tiền xử lý dữ liệu (chuẩn hóa tiền tệ, thời lượng, GPS, text)
    ├── kmeans.py               # Phân cụm K-Means, LSA (Truncated SVD) & TF-IDF
    ├── topsis.py               # Thuật toán xếp hạng đa tiêu chí TOPSIS & tính toán chi phí
    ├── learning.py             # Học máy thích nghi (Logistic Regression & Beta-smoothing)
    ├── evaluation.py           # Đánh giá hệ thống (NDCG@5, Bootstrap, Wilcoxon Test)
    └── feedback_manager.py     # Quản lý đọc/ghi phản hồi (Local Cache + Google Sheets)
```

---

## ⚙️ 3. KIẾN TRÚC & PHƯƠNG PHÁP LUẬN THUẬT TOÁN

Hệ thống hiện thực hóa đầy đủ chu trình 7 bước của một Hệ Trợ giúp Ra Quyết định (DSS):

```
  ┌────────────────┐     ┌────────────────┐     ┌────────────────┐
  │ 1. Người dùng  │ ──> │  2. Thu nhận   │ ──> │ 3. Tiền xử lý  │
  │ nhập yêu cầu   │     │  Sở thích &    │     │  & Phân cụm    │
  │ (Budget, Ngày, │     │  Ràng buộc     │     │  (K-Means+LSA) │
  │  Sở thích,...) │     └────────────────┘     └───────┬────────┘
  └────────────────┘                                    │
                                                        ▼
  ┌────────────────┐     ┌────────────────┐     ┌────────────────┐
  │ 6. Phản hồi    │ <── │ 5. Gợi ý Combo │ <── │ 4. Xếp hạng    │
  │ & Tự học       │     │ Điểm đến + KS  │     │ Đa tiêu chí    │
  │ (Logistic Reg) │     │ + Quán ăn      │     │ (TOPSIS)       │
  └───────┬────────┘     └────────────────┘     └────────────────┘
          │
          └─────────── Cập nhật trọng số & điểm cộng đồng ───────────┘
```

### 3.1. Tiền xử lý dữ liệu (`utils/preprocessing.py`)
* **Chuẩn hóa giá vé & chi phí (`parse_money`):** Xử lý ngôn ngữ tự nhiên về giá (ví dụ: `"500k - 1.5 triệu"`, `"Miễn phí"`, `"Trên 2.5 triệu"`, `"300.000 VNĐ"`) thành cặp giá trị số thực `(cost_min, cost_max)` theo đơn vị đồng nhất VNĐ.
* **Chuẩn hóa thời lượng (`parse_duration`):** Chuyển đổi các định dạng thời gian (`"1-2 ngày"`, `"Nửa ngày"`, `"3 ngày 2 đêm"`) thành số ngày thực tế (ví dụ: `0.5`, `1.0`, `2.0`).
* **Chuẩn hóa điểm đánh giá (`parse_rating`):** Trích xuất điểm số chuẩn thang 5 từ chuỗi `"4.6/5"` hoặc `"4.5/5.0"`.
* **Điền khuyết tọa độ GPS (`COORDS_LOOKUP`):** Tự động tra cứu và điền đầy đủ Vĩ độ (Latitude) và Kinh độ (Longitude) cho toàn bộ 100 điểm đến, làm đầu vào cho mô hình bản đồ PyDeck 3D.

### 3.2. Phân cụm điểm đến bằng K-Means & LSA (`utils/kmeans.py`)
* **Biểu diễn văn bản bằng TF-IDF:** Trích xuất đặc trưng từ khóa kết hợp bộ lọc từ dừng chuyên biệt cho tiếng Việt và ngành du lịch (**`VIETNAMESE_STOP_WORDS`**).
* **Khử nhiễu & giảm chiều LSA (Truncated SVD):** Giảm không gian vector ngữ nghĩa xuống 8 chiều tiềm ẩn, hạn chế hiện tượng thưa thớt của từ khóa.
* **Gom cụm K-Means:** Phân chia các điểm đến thành các cụm có tính đồng nhất cao về chủ đề trải nghiệm và đặc trưng chi phí/thời gian.
* **Tự động gán nhãn chủ đề cụm (Catchy Theme Names):** Định danh các cụm bằng tên gọi hấp dẫn:
  * *Biển đảo hoang sơ*
  * *Núi rừng & Cao nguyên*
  * *Hang động & Kỳ quan đá vôi*
  * *Sông nước & Sinh thái*
  * *Suối thác mộng mơ*
  * *Di tích & Lịch sử*
  * *Kiến trúc & Di sản*
  * *Tâm linh & An yên*
  * *Hiện đại & Giải trí*

### 3.3. Xếp hạng Đa tiêu chí TOPSIS (`utils/topsis.py`)
Hệ thống sử dụng kỹ thuật **TOPSIS (Technique for Order Preference by Similarity to Ideal Solution)** nhằm xác định điểm đến gần giải pháp lý tưởng nhất và xa giải pháp tồi nhất theo 6 tiêu chí:

| Tiêu chí | Ý nghĩa | Loại tiêu chí | Trọng số khởi điểm (Có query / Không query) |
| :--- | :--- | :---: | :---: |
| **$C_1$: rating_num** | Điểm đánh giá trung bình của điểm đến | Benefit | 0.15 / 0.32 |
| **$C_2$: budget_fit** | Độ tương thích giữa chi phí thực tế và ngân sách cả nhóm | Benefit | 0.13 / 0.30 |
| **$C_3$: duration_fit** | Độ tương thích với thời lượng mong muốn của du khách | Benefit | 0.08 / 0.20 |
| **$C_4$: style_match** | Độ tương đồng Cosine Similarity giữa câu hỏi và từ khóa | Benefit | 0.40 / 0.04 |
| **$C_5$: cluster_fit** | Độ tương đồng ngữ nghĩa tới tâm cụm K-Means | Benefit | 0.16 / 0.04 |
| **$C_6$: community_score** | Điểm tín nhiệm học từ đánh giá cộng đồng du khách | Benefit | 0.08 / 0.10 |

**Thuật toán TOPSIS chi tiết:**
1. **Lập ma trận quyết định:** $X = [x_{ij}]_{m \times n}$ với $m$ điểm đến và $n = 6$ tiêu chí.
2. **Chuẩn hóa vector ma trận quyết định:**
   $$r_{ij} = \frac{x_{ij}}{\sqrt{\sum_{k=1}^m x_{kj}^2}}$$
3. **Nhân trọng số tiêu chí:** $v_{ij} = w_j \cdot r_{ij}$, với $\sum_{j=1}^n w_j = 1$.
4. **Xác định giải pháp lý tưởng dương ($A^+$) và âm ($A^-$):**
   $$A^+ = \{ \max_i v_{ij} \mid j \in J \}, \quad A^- = \{ \min_i v_{ij} \mid j \in J \}$$
5. **Tính khoảng cách Euclidean:**
   $$S_i^+ = \sqrt{\sum_{j=1}^n (v_{ij} - v_j^+)^2}, \quad S_i^- = \sqrt{\sum_{j=1}^n (v_{ij} - v_j^-)^2}$$
6. **Tính hệ số tương tự giải pháp lý tưởng ($C_i^*$):**
   $$C_i^* = \frac{S_i^-}{S_i^+ + S_i^-} \quad (0 \le C_i^* \le 1)$$
   Điểm số $C_i^*$ càng cao thể hiện điểm đến càng phù hợp với yêu cầu của du khách.

### 3.4. Cơ chế Học máy Thích nghi từ Phản hồi (`utils/learning.py`)
* **Điểm tín nhiệm cộng đồng (Beta-smoothing):** Áp dụng ước lượng Bayes phân phối Beta(1,1):
  $$\text{community\_score} = \frac{\text{fit} + 1}{\text{fit} + \text{unfit} + 2}$$
  Khi một điểm đến chưa có lượt đánh giá nào, điểm mặc định đạt $0.5$ (trung lập).
* **Tự động cập nhật trọng số bằng Logistic Regression:**
  Khi người dùng gửi phản hồi tường minh (Top-5 điểm đến: *Phù hợp* hoặc *Chưa phù hợp*), mô hình Logistic Regression sẽ học xác suất người dùng chấp nhận điểm đến từ các đặc trưng TOPSIS. Trọng số mới được hòa trộn với trọng số ban đầu qua hệ số co (Shrinkage factor):
  $$\lambda = \frac{N}{N + 100}, \quad W_{\text{new}} = (1 - \lambda) \cdot W_{\text{init}} + \lambda \cdot W_{\text{learned}}$$
  *(Ít dữ liệu -> ưu tiên trọng số chuyên gia; Nhiều dữ liệu -> trọng số thích nghi theo thị hiếu người dùng).*

### 3.5. Đánh giá chất lượng hệ thống (`utils/evaluation.py`)
* **Đánh giá Offline trên 140 kịch bản du lịch độc lập:**
  * So sánh với 3 mô hình Baseline: *Ngẫu nhiên (Random)*, *Chỉ xếp theo Rating*, *Chỉ TF-IDF Content-based*.
  * Chỉ số đo lường: **NDCG@5**, **Precision@5**, **Tỷ lệ đúng chủ đề**, **Tỷ lệ trong ngân sách**, **Đa dạng cụm**, **Độ phủ danh mục**.
  * Ước lượng khoảng tin cậy **Bootstrap 95% CI** (1,000 lần lặp).
  * Kiểm định cặp đôi phi tham số **Wilcoxon Signed-Rank Test** khẳng định mức độ vượt trội của hệ thống DSS có ý nghĩa thống kê ($p < 0.001$).
* **Đánh giá Online:** Thu thập trực tiếp từ hành vi tương tác thực tế của du khách:
  * **Precision@5:** Tỷ lệ điểm đến được gắn nhãn "Phù hợp" trong top-5 gợi ý.
  * **Hit-rate@5:** Tỷ lệ phiên gợi ý có ít nhất một điểm đến phù hợp.
  * **Số sao đánh giá trung bình** của người dùng.

---

## 🚀 4. HƯỚNG DẪN CÀI ĐẶT & CHẠY ỨNG DỤNG CỤC BỘ (LOCAL)

### 4.1. Yêu cầu môi trường
* Hệ điều hành: Windows, macOS hoặc Linux.
* Python: **Phiên bản 3.9 đến 3.12**.

### 4.2. Các bước cài đặt

**Bước 1: Tải mã nguồn về máy và mở Terminal / PowerShell tại thư mục dự án:**
```bash
cd "duong_dan_thu_muc_DSS"
```

**Bước 2: (Khuyến nghị) Tạo và kích hoạt môi trường ảo:**
```bash
# Trên Windows:
python -m venv venv
.\venv\Scripts\activate

# Trên macOS/Linux:
python3 -m venv venv
source venv/bin/activate
```

**Bước 3: Cài đặt đầy đủ các thư viện phụ thuộc:**
```bash
pip install -r requirements.txt
```

### 4.3. Khởi chạy ứng dụng

* **Cách 1 (Nhanh nhất trên Windows):** Click đúp vào file **`run.bat`**.
* **Cách 2 (Dùng lệnh):**
  ```bash
  streamlit run app.py
  ```

Trình duyệt sẽ tự động mở giao diện tại: `http://localhost:8501`.

### 4.4. Chạy kiểm thử Tiền xử lý dữ liệu (Tùy chọn)
Nếu muốn tái tạo lại các file dữ liệu làm sạch từ dữ liệu gốc, chạy lệnh:
```bash
python utils/preprocessing.py
```

---

## 🌐 5. CẤU HÌNH CLOUD WEBHOOK GOOGLE SHEETS

Hệ thống mặc định hoạt động ngoại tuyến (lưu tại [`data/feedback_store.json`](file:///c:/Users/Lenovo/Downloads/DSS%20%281%29/DSS/data/feedback_store.json)). Nếu muốn tự triển khai Backend Google Sheets của riêng bạn:
1. Tạo một trang tính mới tại [Google Sheets](https://sheets.new).
2. Chọn **Tiện ích mở rộng (Extensions)** -> **Apps Script**.
3. Dán toàn bộ mã nguồn từ file [`google_sheets_script.js`](file:///c:/Users/Lenovo/Downloads/DSS%20%281%29/DSS/google_sheets_script.js) vào trình biên soạn.
4. Bấm **Triển khai (Deploy)** -> **Tùy chọn triển khai mới (New deployment)** -> Chọn loại **Ứng dụng web (Web app)**:
   * *Thực thi dưới dạng:* **Tôi (tài khoản cá nhân)**.
   * *Ai có quyền truy cập:* **Bất kỳ ai (Anyone)**.
5. Sao chép URL ứng dụng web (đuôi `/exec`) và dán vào [`.streamlit/secrets.toml`](file:///c:/Users/Lenovo/Downloads/DSS%20%281%29/DSS/.streamlit/secrets.toml):
   ```toml
   FEEDBACK_WEBHOOK_URL = "URL_APPS_SCRIPT_CUA_BAN"
   ```

---

## 📦 6. HƯỚNG DẪN ĐÓNG GÓI NỘP BÀI THI (SUBMISSION CHECKLIST)

Khi nén bài thi để nộp cho Giảng viên hoặc tải lên cổng đào tạo (LMS / Moodle / Teams):

1. **Lệnh tạo file ZIP tự động chuẩn hóa bằng PowerShell:**
   ```powershell
   Compress-Archive -Path "app.py", "run.bat", "requirements.txt", "README.md", "google_sheets_script.js", ".streamlit", "data", "utils" -DestinationPath "..\BaiThi_DSS_DuLich.zip" -Force
   ```

2. **Bảng kiểm tra (Checklist):**
   - [x] Đã cập nhật đầy đủ thông tin 5 thành viên nhóm và tỉ lệ đóng góp.
   - [x] Đã kèm link demo sản phẩm đã deploy: [https://travel-dss.streamlit.app/](https://travel-dss.streamlit.app/).
   - [x] Ứng dụng chạy mượt mà, không phát sinh lỗi ngoại lệ.
   - [x] Đầy đủ dữ liệu thô và dữ liệu làm sạch trong thư mục `data/`.
   - [x] Kèm file `run.bat` giúp giảng viên mở app bằng 1 click chuột.
   - [x] File ZIP không bị dính rác `__pycache__`, `venv` hay `.git`.

---
*Kính chúc Quý Thầy/Cô có trải nghiệm đánh giá tốt với sản phẩm của nhóm!*
