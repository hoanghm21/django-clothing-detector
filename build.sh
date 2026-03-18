#!/usr/bin/env bash
# Thoát ngay nếu có lỗi
set -o errexit

# Cài đặt thư viện (production dùng file nhẹ hơn)
if [ -f requirements-prod.txt ]; then
    pip install -r requirements-prod.txt
else
    pip install -r requirements.txt
fi

# Gom các file static (CSS, JS, Images) để Render phục vụ
python manage.py collectstatic --no-input

# Cập nhật cấu hình Database (Migrate)
python manage.py migrate