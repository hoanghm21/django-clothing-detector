"""
YOLOv8 DeepFashion2 Clothing Detection
13 Categories với tiếng Anh (như yêu cầu bài tập)
"""
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
import requests
from django.conf import settings

# DeepFashion2 Categories (13 classes)
DEEPFASHION2_CATEGORIES = {
    0: 'short_sleeved_shirt',
    1: 'long_sleeved_shirt', 
    2: 'short_sleeved_outwear',
    3: 'long_sleeved_outwear',
    4: 'vest',
    5: 'sling',
    6: 'shorts',
    7: 'trousers',
    8: 'skirt',
    9: 'short_sleeved_dress',
    10: 'long_sleeved_dress',
    11: 'vest_dress',
    12: 'sling_dress'
}

# Màu cho mỗi category
COLOR_PALETTE = {
    0: (0, 255, 0),      # short_sleeved_shirt - xanh lá
    1: (0, 200, 0),      # long_sleeved_shirt - xanh lá đậm
    2: (255, 165, 0),    # short_sleeved_outwear - cam
    3: (255, 140, 0),    # long_sleeved_outwear - cam đậm
    4: (0, 191, 255),    # vest - xanh dương
    5: (255, 105, 180),  # sling - hồng
    6: (0, 0, 255),      # shorts - xanh dương đậm
    7: (65, 105, 225),   # trousers - xanh hoàng gia
    8: (255, 0, 255),    # skirt - tím hồng
    9: (148, 0, 211),    # short_sleeved_dress - tím
    10: (138, 43, 226),  # long_sleeved_dress - tím xanh
    11: (75, 0, 130),    # vest_dress - indigo
    12: (199, 21, 133),  # sling_dress - hồng đậm
}

# Global model cache
_model = None

# Check if running in production (Render)
def is_production():
    return not settings.DEBUG


def get_model():
    """Load YOLOv8 DeepFashion2 model từ Hugging Face (chỉ local)"""
    global _model
    
    if _model is None:
        from ultralytics import YOLO
        from huggingface_hub import hf_hub_download
        
        print("🔄 Đang tải YOLOv8 DeepFashion2 model từ Hugging Face...")
        
        # Tải model từ Hugging Face
        model_path = hf_hub_download(
            repo_id="Bingsu/adetailer",
            filename="deepfashion2_yolov8s-seg.pt"
        )
        
        _model = YOLO(model_path)
        print("✅ YOLOv8 DeepFashion2 model đã sẵn sàng!")
    
    return _model


def detect_clothing_api(image_path, confidence=0.25):
    """
    Gọi HuggingFace Inference API (dùng cho production).
    Không cần load model vào RAM.
    """
    API_URL = "https://api-inference.huggingface.co/models/Bingsu/adetailer"
    
    # Đọc file ảnh
    with open(image_path, "rb") as f:
        image_data = f.read()
    
    # Gọi API
    response = requests.post(API_URL, data=image_data, timeout=60)
    
    if response.status_code != 200:
        print(f"API Error: {response.status_code} - {response.text}")
        return []
    
    result = response.json()
    
    # Parse kết quả từ API
    clothing_items = []
    
    if isinstance(result, list):
        for detection in result:
            label = detection.get('label', '')
            score = detection.get('score', 0)
            box = detection.get('box', {})
            
            if score < confidence:
                continue
            
            # Map label to category
            label_id = None
            for id, name in DEEPFASHION2_CATEGORIES.items():
                if name in label.lower():
                    label_id = id
                    break
            
            if label_id is None:
                label_id = 0
            
            clothing_items.append({
                'label': DEEPFASHION2_CATEGORIES.get(label_id, label),
                'label_vn': DEEPFASHION2_CATEGORIES.get(label_id, label),
                'score': score,
                'bbox': [box.get('xmin', 0), box.get('ymin', 0), box.get('xmax', 0), box.get('ymax', 0)],
                'label_id': label_id
            })
    
    return clothing_items


def detect_clothing_local(image_path, confidence=0.25):
    """
    Chạy model local (dùng cho development).
    """
    model = get_model()
    
    # Run inference
    results = model(image_path, verbose=False)
    
    clothing_items = []
    
    for result in results:
        boxes = result.boxes
        masks = result.masks  # Segmentation masks
        
        if boxes is None:
            continue
            
        for i in range(len(boxes)):
            cls = int(boxes.cls[i])
            conf = float(boxes.conf[i])
            
            if conf < confidence:
                continue
            
            # Lấy tên category từ model (DeepFashion2 categories)
            label = DEEPFASHION2_CATEGORIES.get(cls, f'class_{cls}')
            
            # Lấy bounding box
            x1, y1, x2, y2 = boxes.xyxy[i].tolist()
            
            item = {
                'label': label,
                'label_vn': label,  # Giữ tiếng Anh
                'score': conf,
                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                'label_id': cls
            }
            
            # Lấy polygon coordinates nếu có masks
            if masks is not None and i < len(masks):
                # masks.xy chứa polygon coordinates
                if hasattr(masks, 'xy') and i < len(masks.xy):
                    item['polygon'] = masks.xy[i]  # numpy array of (x, y) points
            
            clothing_items.append(item)
    
    return clothing_items


def detect_clothing(image_path, confidence=0.25):
    """
    Nhận diện quần áo trong ảnh.
    - Production (Render): Gọi HuggingFace API
    - Development (Local): Load model local
    
    Returns:
        list: Danh sách các item được phát hiện với label tiếng Anh
    """
    if is_production():
        print("🌐 Using HuggingFace Inference API (production mode)")
        return detect_clothing_api(image_path, confidence)
    else:
        print("💻 Using local model (development mode)")
        return detect_clothing_local(image_path, confidence)


def create_polygon_image(original_path, clothing_items, output_path):
    """
    Tạo ảnh với polygon tô màu:
    - Viền đậm màu xung quanh
    - Phía trong nhạt hơn (semi-transparent fill)
    """
    import cv2
    from PIL import ImageOps
    
    # Mở ảnh gốc và xử lý EXIF orientation
    original = Image.open(original_path).convert('RGB')
    original = ImageOps.exif_transpose(original)  # Fix rotation từ EXIF
    img = np.array(original)
    
    # Tạo overlay layer cho transparent fill
    overlay = img.copy()
    
    # Load font
    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except:
        font = ImageFont.load_default()
    
    label_positions = []  # Lưu vị trí để vẽ label sau
    
    for item in clothing_items:
        label = item.get('label', 'item')
        score = item.get('score', 0)
        label_id = item.get('label_id', 0)
        
        # Chọn màu (BGR for OpenCV)
        color = COLOR_PALETTE.get(label_id, (0, 255, 0))
        color_bgr = (color[2], color[1], color[0])
        
        polygon = item.get('polygon')
        bbox = item.get('bbox', [])
        
        if polygon is not None and len(polygon) > 2:
            # Vẽ filled polygon
            pts = np.array(polygon, dtype=np.int32)
            
            # Fill với màu nhạt (trên overlay)
            cv2.fillPoly(overlay, [pts], color_bgr)
            
            # Vẽ viền đậm (trên ảnh gốc)
            cv2.polylines(img, [pts], True, color_bgr, 2)
            
            # Lấy vị trí để vẽ label
            x, y = int(pts[:, 0].min()), int(pts[:, 1].min())
            label_positions.append((x, y, label, score, color))
            
        elif len(bbox) >= 4:
            # Fallback: vẽ bounding box nếu không có polygon
            x1, y1, x2, y2 = [int(v) for v in bbox]
            
            # Fill với màu nhạt
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color_bgr, -1)
            
            # Vẽ viền đậm
            cv2.rectangle(img, (x1, y1), (x2, y2), color_bgr, 2)
            
            label_positions.append((x1, y1, label, score, color))
    
    # Blend overlay với ảnh gốc (alpha=0.3 cho fill nhạt)
    alpha = 0.3
    img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
    
    # Chuyển sang Pillow để vẽ text
    result = Image.fromarray(img)
    draw = ImageDraw.Draw(result)
    
    # Vẽ labels
    for x, y, label, score, color in label_positions:
        label_text = f"{label} {score:.2f}"
        text_y = max(0, y - 18)
        text_bbox = draw.textbbox((x, text_y), label_text, font=font)
        draw.rectangle([text_bbox[0]-2, text_bbox[1]-2, text_bbox[2]+2, text_bbox[3]+2], fill=color)
        draw.text((x, text_y), label_text, fill='white', font=font)
    
    result.save(output_path)
    return output_path


def crop_clothing_items(original_path, clothing_items, unique_name):
    """Cắt riêng từng item trang phục"""
    from django.conf import settings
    from PIL import ImageOps
    
    original = Image.open(original_path).convert('RGB')
    original = ImageOps.exif_transpose(original)  # Fix rotation từ EXIF
    cropped_items = []
    
    crops_dir = os.path.join(settings.MEDIA_ROOT, 'crops')
    os.makedirs(crops_dir, exist_ok=True)
    
    for i, item in enumerate(clothing_items):
        bbox = item.get('bbox', [])
        if len(bbox) < 4:
            continue
            
        x1, y1, x2, y2 = bbox
        label = item.get('label', 'item')
        
        # Crop ảnh
        cropped = original.crop((x1, y1, x2, y2))
        
        # Lưu
        crop_filename = f'{unique_name}_{i}.png'
        crop_path = os.path.join(crops_dir, crop_filename)
        cropped.save(crop_path)
        
        cropped_items.append({
            'url': settings.MEDIA_URL + f'crops/{crop_filename}',
            'label_vn': label,  # Giữ tiếng Anh theo yêu cầu
        })
    
    return cropped_items
