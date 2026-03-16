# Django Clothing Detector

Ứng dụng Django để phát hiện trang phục sử dụng YOLOv8 DeepFashion2 model.

## 🚀 Tính năng

- **Clothing Detection**: Phát hiện 13 loại trang phục trong ảnh
- **Polls App**: Ứng dụng bình chọn cơ bản Django
- **REST API**: API endpoint để tích hợp với các ứng dụng khác

### 13 Loại trang phục được hỗ trợ:
- short_sleeved_shirt, long_sleeved_shirt
- short_sleeved_outwear, long_sleeved_outwear
- vest, sling, shorts, trousers, skirt
- short_sleeved_dress, long_sleeved_dress
- vest_dress, sling_dress

## 📋 Yêu cầu hệ thống

- Python 3.11+
- pip

## 🛠️ Cài đặt Local

### 1. Clone repository
```bash
git clone https://github.com/YOUR_USERNAME/djangotutorial.git
cd djangotutorial
```

### 2. Tạo virtual environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/macOS
source venv/bin/activate
```

### 3. Cài đặt dependencies
```bash
pip install -r requirements.txt
```

### 4. Cấu hình environment
```bash
# Copy file env mẫu
copy .env.example .env   # Windows
cp .env.example .env     # Linux/macOS

# Chỉnh sửa .env theo nhu cầu
```

### 5. Chạy migrations
```bash
python manage.py migrate
```

### 6. Tạo superuser (tùy chọn)
```bash
python manage.py createsuperuser
```

### 7. Chạy server
```bash
python manage.py runserver
```

Truy cập:
- Home: http://127.0.0.1:8000/
- Clothing Detector: http://127.0.0.1:8000/clothing/
- Polls: http://127.0.0.1:8000/polls/
- Admin: http://127.0.0.1:8000/admin/

## 📡 API Endpoint

### POST /clothing/api/upload/

Upload ảnh để phát hiện trang phục.

**Request:**
```bash
curl -X POST -F "image=@your_image.jpg" http://127.0.0.1:8000/clothing/api/upload/
```

**Response:**
```json
{
    "success": true,
    "original_url": "/media/uploads/xxx.jpg",
    "result_url": "/media/results/xxx_polygon.png",
    "detections": [
        {
            "label": "short_sleeved_shirt",
            "score": 0.95,
            "bbox": [100, 50, 300, 400]
        }
    ],
    "cropped_items": ["/media/crops/xxx_0.png"]
}
```

## 🌐 Deploy lên Production

### Heroku
```bash
# Login Heroku
heroku login

# Tạo app
heroku create your-app-name

# Deploy
git push heroku main

# Chạy migrations
heroku run python manage.py migrate
```

### Railway / Render
1. Kết nối GitHub repository
2. Cấu hình environment variables
3. Deploy tự động

## 📁 Cấu trúc Project

```
djangotutorial/
├── clothing_detector/     # App phát hiện trang phục
│   ├── detector_yolo.py   # YOLOv8 model
│   ├── views.py           # Views & API
│   └── templates/         # HTML templates
├── polls/                 # App bình chọn
├── mysite/                # Django settings
├── media/                 # User uploads
├── templates/             # Global templates
├── manage.py
├── requirements.txt
├── Procfile               # Heroku config
└── runtime.txt            # Python version
```

## 🔧 Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| SECRET_KEY | Django secret key | - |
| DEBUG | Debug mode | True |
| ALLOWED_HOSTS | Allowed hosts | localhost |
| DATABASE_URL | Database URL (production) | SQLite |

## 📝 License

MIT License

## 👤 Author

Your Name
