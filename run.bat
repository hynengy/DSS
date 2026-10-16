@echo off
chcp 65001 >nul
title Hệ thống Trợ giúp Ra quyết định Du lịch Việt Nam (DSS)
echo ======================================================================
echo       HỆ THỐNG TRỢ GIÚP RA QUYẾT ĐỊNH DU LỊCH VIỆT NAM (DSS TRAVEL)
echo ======================================================================
echo.
echo [*] Đang kiểm tra môi trường và khởi động ứng dụng Streamlit...
echo [*] Trình duyệt sẽ tự động mở tại http://localhost:8501
echo.
python -m streamlit run app.py
if errorlevel 1 (
    echo.
    echo [!] Có lỗi xảy ra khi khởi động. Hãy đảm bảo bạn đã cài đặt các thư viện:
    echo     pip install -r requirements.txt
    echo.
    pause
)
