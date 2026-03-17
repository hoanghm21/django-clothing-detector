#!/usr/bin/env bash
# Thoát ngay nếu có lỗi
set -o errexit

# Cài đặt thư viện
pip install -r requirements.txt

# Gom các file static (CSS, JS, Images) để Render phục vụ
python manage.py collectstatic --no-input

# Cập nhật cấu hình Database (Migrate)
python manage.py migrate