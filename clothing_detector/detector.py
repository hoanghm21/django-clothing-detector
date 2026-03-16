from transformers import pipeline
from PIL import Image
import numpy as np
import os

# Palette màu cố định cho mỗi loại quần áo (RGB)
COLOR_PALETTE = {
    0: (0, 0, 0),        # Background - đen (trong suốt)
    1: (255, 0, 0),      # Hat - đỏ
    2: (0, 0, 0),        # Hair - ẩn
    3: (255, 165, 0),    # Sunglasses - cam
    4: (0, 128, 255),    # Upper-clothes - xanh dương
    5: (255, 0, 255),    # Skirt - hồng
    6: (0, 255, 0),      # Pants - xanh lá
    7: (128, 0, 128),    # Dress - tím
    8: (255, 255, 0),    # Belt - vàng
    9: (0, 255, 255),    # Left-shoe - cyan
    10: (0, 255, 255),   # Right-shoe - cyan
    11: (0, 0, 0),       # Face - ẩn
    12: (0, 0, 0),       # Left-leg - ẩn
    13: (0, 0, 0),       # Right-leg - ẩn
    14: (0, 0, 0),       # Left-arm - ẩn
    15: (0, 0, 0),       # Right-arm - ẩn
    16: (165, 42, 42),   # Bag - nâu
    17: (255, 192, 203), # Scarf - hồng nhạt
}

# Tên tiếng Việt cho các labels
LABEL_NAMES = {
    0: "Nền", 1: "Mũ", 2: "Tóc", 3: "Kính mát",
    4: "Áo", 5: "Váy", 6: "Quần", 7: "Đầm",
    8: "Thắt lưng", 9: "Giày trái", 10: "Giày phải",
    11: "Mặt", 12: "Chân trái", 13: "Chân phải",
    14: "Tay trái", 15: "Tay phải", 16: "Túi", 17: "Khăn"
}

# Các labels cần hiển thị (quần áo + phụ kiện)
CLOTHING_LABELS = [1, 3, 4, 5, 6, 7, 8, 9, 10, 16, 17]

# Global variable để cache model
_segmenter = None

def get_segmenter():
    """Load model (chỉ load 1 lần)"""
    global _segmenter
    if _segmenter is None:
        print("🔄 Đang tải model... (lần đầu sẽ mất vài phút)")
        _segmenter = pipeline("image-segmentation", model="mattmdjaga/segformer_b2_clothes")
        print("✅ Model đã sẵn sàng!")
    return _segmenter

def detect_clothing(image_path):
    """
    Nhận diện quần áo trong ảnh
    
    Returns:
        results: list các item được nhận diện
    """
    segmenter = get_segmenter()

    # Chạy inference
    results = segmenter(image_path)

    # Mapping từ label tiếng Anh sang ID
    LABEL_TO_ID = {
        'Background': 0, 'Hat': 1, 'Hair': 2, 'Sunglasses': 3,
        'Upper-clothes': 4, 'Skirt': 5, 'Pants': 6, 'Dress': 7,
        'Belt': 8, 'Left-shoe': 9, 'Right-shoe': 10, 'Face': 11,
        'Left-leg': 12, 'Right-leg': 13, 'Left-arm': 14, 'Right-arm': 15,
        'Bag': 16, 'Scarf': 17
    }

    # Lọc chỉ lấy quần áo và phụ kiện
    clothing_items = []
    has_upper = False
    has_pants = False
    
    for item in results:
        label = item['label']
        label_id = LABEL_TO_ID.get(label, -1)
        
        if label_id == 4:  # Upper-clothes
            has_upper = True
        if label_id == 6:  # Pants
            has_pants = True
        
        if label_id in CLOTHING_LABELS:
            clothing_items.append({
                'label': label,
                'label_vn': LABEL_NAMES.get(label_id, label),
                'score': item.get('score', 1.0),
                'mask': item['mask'],
                'label_id': label_id
            })
    
    # NOTE: Đã tắt heuristics vì gây nhận nhầm - đầm thật bị tách thành áo+quần
    # Giữ nguyên label từ model
    # clothing_items = apply_dress_heuristics(clothing_items, has_upper, has_pants)

    return clothing_items


def apply_dress_heuristics(clothing_items, has_upper, has_pants):
    """
    Áp dụng heuristics để xử lý trường hợp model nhận nhầm Áo+Quần thành Đầm.
    
    Nếu:
    - Có Dress (Đầm) được detect
    - Không có Upper-clothes hoặc Pants riêng biệt
    - Mask của Dress bao phủ >60% chiều cao ảnh
    
    Thì: Có khả năng đây là Áo + Quần, cần tách đôi
    """
    new_items = []
    
    for item in clothing_items:
        if item['label_id'] == 7:  # Dress
            mask = item['mask']
            mask_array = np.array(mask)
            
            # Tìm bounding box của mask
            rows = np.any(mask_array > 128, axis=1)
            if not np.any(rows):
                new_items.append(item)
                continue
            
            y_indices = np.where(rows)[0]
            y_min, y_max = y_indices[0], y_indices[-1]
            mask_height_ratio = (y_max - y_min) / mask_array.shape[0]
            
            # Nếu mask chiếm >55% chiều cao và không có upper/pants riêng
            if mask_height_ratio > 0.55 and not (has_upper and has_pants):
                # Tách thành Áo và Quần
                mid_y = (y_min + y_max) // 2
                
                # Tạo mask cho Áo (phần trên)
                upper_mask_array = mask_array.copy()
                upper_mask_array[mid_y:, :] = 0
                upper_mask = Image.fromarray(upper_mask_array)
                
                # Tạo mask cho Quần (phần dưới)
                lower_mask_array = mask_array.copy()
                lower_mask_array[:mid_y, :] = 0
                lower_mask = Image.fromarray(lower_mask_array)
                
                # Thêm Áo
                new_items.append({
                    'label': 'Upper-clothes',
                    'label_vn': 'Áo',
                    'score': item['score'],
                    'mask': upper_mask,
                    'label_id': 4
                })
                
                # Thêm Quần
                new_items.append({
                    'label': 'Pants',
                    'label_vn': 'Quần',
                    'score': item['score'],
                    'mask': lower_mask,
                    'label_id': 6
                })
            else:
                # Giữ nguyên là Đầm
                new_items.append(item)
        else:
            new_items.append(item)
    
    return new_items

def create_overlay_image(original_path, clothing_items, output_path):
    """Tạo ảnh với mask overlay"""
    # Mở ảnh gốc
    original = Image.open(original_path).convert('RGBA')

    # Tạo layer overlay
    overlay = Image.new('RGBA', original.size, (0, 0, 0, 0))

    for item in clothing_items:
        mask = item['mask']
        label_id = item.get('label_id', 0)
        color = COLOR_PALETTE.get(label_id, (128, 128, 128))

        # Resize mask về kích thước ảnh gốc
        mask = mask.resize(original.size, Image.NEAREST)
        mask_array = np.array(mask)
        
        # Tạo colored mask
        colored = Image.new('RGBA', original.size, (*color, 100)) # Alpha = 100

        # Alpha mask
        mask_bool = mask_array > 128
        overlay_array = np.array(overlay)
        colored_array = np.array(colored)

        for c in range(4):
            overlay_array[:, :, c] = np.where(mask_bool, colored_array[:, :, c], overlay_array[:, :, c])
        overlay = Image.fromarray(overlay_array)

    # Merge với ảnh gốc
    result = Image.alpha_composite(original, overlay)
    result = result.convert('RGB')
    result.save(output_path)

    return output_path


def create_bbox_image(original_path, clothing_items, output_path):
    """
    Tạo ảnh với bounding box quanh trang phục (yêu cầu bài tập).
    Vẽ khung chữ nhật màu + label cho từng item.
    """
    from PIL import ImageDraw, ImageFont
    
    # Mở ảnh gốc
    original = Image.open(original_path).convert('RGB')
    draw = ImageDraw.Draw(original)
    
    # Font cho label (dùng font mặc định)
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except:
        font = ImageFont.load_default()
    
    for item in clothing_items:
        mask = item['mask']
        label_id = item.get('label_id', 0)
        label_vn = item.get('label_vn', 'Item')
        color = COLOR_PALETTE.get(label_id, (255, 0, 0))
        
        # Resize mask về kích thước ảnh
        mask = mask.resize(original.size, Image.NEAREST)
        mask_array = np.array(mask)
        
        # Tìm bounding box từ mask
        rows = np.any(mask_array > 128, axis=1)
        cols = np.any(mask_array > 128, axis=0)
        
        if not np.any(rows) or not np.any(cols):
            continue
        
        y_min, y_max = np.where(rows)[0][[0, -1]]
        x_min, x_max = np.where(cols)[0][[0, -1]]
        
        # Vẽ khung chữ nhật
        box_coords = [x_min, y_min, x_max, y_max]
        draw.rectangle(box_coords, outline=color, width=3)
        
        # Vẽ label background
        text_bbox = draw.textbbox((x_min, y_min - 20), label_vn, font=font)
        draw.rectangle([text_bbox[0]-2, text_bbox[1]-2, text_bbox[2]+2, text_bbox[3]+2], fill=color)
        
        # Vẽ text label
        draw.text((x_min, y_min - 20), label_vn, fill='white', font=font)
    
    original.save(output_path)
    return output_path


def create_polygon_image(original_path, clothing_items, output_path):
    """
    Tạo ảnh với đường viền polygon sát theo hình dạng trang phục.
    Sử dụng Pillow để vẽ text (hỗ trợ tiếng Việt).
    """
    import cv2
    from PIL import ImageDraw, ImageFont
    
    # Mở ảnh gốc
    original = Image.open(original_path).convert('RGB')
    img_cv = np.array(original)
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)
    
    # Vẽ contours bằng OpenCV
    label_positions = []  # Lưu vị trí để vẽ label sau
    
    for item in clothing_items:
        mask = item['mask']
        label_id = item.get('label_id', 0)
        label_vn = item.get('label_vn', 'Item')
        color = COLOR_PALETTE.get(label_id, (255, 0, 0))
        color_bgr = (color[2], color[1], color[0])
        
        # Resize mask về kích thước ảnh
        mask = mask.resize(original.size, Image.NEAREST)
        mask_array = np.array(mask)
        
        # Chuyển mask thành binary
        binary_mask = (mask_array > 128).astype(np.uint8) * 255
        
        # Tìm contours
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            continue
        
        # Vẽ contours
        cv2.drawContours(img_cv, contours, -1, color_bgr, 2)
        
        # Lưu vị trí để vẽ label
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        label_positions.append((x, y, label_vn, color))
    
    # Chuyển lại sang RGB/Pillow để vẽ text tiếng Việt
    img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    result = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(result)
    
    # Load font hỗ trợ tiếng Việt
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except:
        font = ImageFont.load_default()
    
    # Vẽ labels bằng Pillow (hỗ trợ tiếng Việt)
    for x, y, label_vn, color in label_positions:
        text_y = max(0, y - 22)
        
        # Background cho text
        text_bbox = draw.textbbox((x, text_y), label_vn, font=font)
        draw.rectangle([text_bbox[0]-2, text_bbox[1]-2, text_bbox[2]+2, text_bbox[3]+2], fill=color)
        
        # Vẽ text
        draw.text((x, text_y), label_vn, fill='white', font=font)
    
    result.save(output_path)
    return output_path


def crop_clothing_items(original_path, clothing_items, unique_name):
    """Cắt riêng từng item quần áo"""
    from django.conf import settings
    
    original = Image.open(original_path).convert('RGBA')
    cropped_items = []
    
    # Tạo thư mục crops
    crops_dir = os.path.join(settings.MEDIA_ROOT, 'crops')
    os.makedirs(crops_dir, exist_ok=True)
    
    for i, item in enumerate(clothing_items):
        mask = item['mask']
        label_id = item.get('label_id', 0)
        label_vn = item.get('label_vn', 'item')

        # Resize mask về kích thước ảnh gốc
        mask = mask.resize(original.size, Image.NEAREST)
        mask_array = np.array(mask)

        # Tìm bouding box của mask
        rows = np.any(mask_array > 128, axis=1)
        cols = np.any(mask_array > 128, axis=0)

        if not np.any(rows) or not np.any(cols):
            continue

        # Tính toán bounding box
        y_min, y_max = np.where(rows)[0][[0, -1]]
        x_min, x_max = np.where(cols)[0][[0, -1]]

        # Tạo ảnh crop với background trong suốt
        cropped = Image.new('RGBA', (x_max - x_min + 1, y_max - y_min + 1), (0, 0, 0, 0))
        
        # Copy pixels từ ảnh gốc theo mask
        for y in range(y_min, y_max + 1):
            for x in range(x_min, x_max + 1):
                if mask_array[y, x] > 128:
                    pixel = original.getpixel((x, y))
                    cropped.putpixel((x - x_min, y - y_min), pixel)
        
        # Lưu ảnh crop
        crop_filename = f'{unique_name}_{label_id}_{i}.png'
        crop_path = os.path.join(crops_dir, crop_filename)
        cropped.save(crop_path)
        
        cropped_items.append({
            'url': settings.MEDIA_URL + f'crops/{crop_filename}',
            'label_vn': label_vn,
            'label_id': label_id,
        })

    return cropped_items
        