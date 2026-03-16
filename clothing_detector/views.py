from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from django.conf import settings
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import os
import uuid
import numpy as np
from PIL import Image

# Sử dụng YOLOv8 DeepFashion2 model (13 categories tiếng Anh)
from .detector_yolo import detect_clothing, create_polygon_image, crop_clothing_items

RECOMMENDED_MAX_IMAGES = 5  # Khuyến nghị tối đa (không phải giới hạn cứng)


def upload_image(request):
    """
    View xử lý upload ảnh và phát hiện trang phục (render HTML).
    Hỗ trợ upload nhiều ảnh (khuyến nghị tối đa 5 ảnh).
    """
    context = {'recommended_max': RECOMMENDED_MAX_IMAGES}

    if request.method == "POST" and request.FILES.getlist('image'):
        images = request.FILES.getlist('image')  # Không giới hạn, xử lý tất cả
        results = []
        
        for image in images:
            result = process_image(image)
            results.append(result)
        
        context['results'] = results

    return render(request, 'clothing_detector/upload.html', context)


@csrf_exempt
def api_upload(request):
    """
    API endpoint: POST /upload
    
    Request:
        - Method: POST
        - Content-Type: multipart/form-data
        - Body: image file (field name: 'image')
    
    Response (JSON):
        {
            "success": true,
            "original_url": "/media/uploads/xxx.jpg",
            "result_url": "/media/results/xxx_polygon.png",
            "detections": [
                {
                    "label": "short_sleeved_shirt",
                    "score": 0.95,
                    "bbox": [x1, y1, x2, y2]
                }
            ]
        }
    """
    if request.method != "POST":
        return JsonResponse({
            'success': False,
            'error': 'Only POST method allowed'
        }, status=405)
    
    if not request.FILES.get('image'):
        return JsonResponse({
            'success': False,
            'error': 'No image file provided. Use field name "image"'
        }, status=400)
    
    image = request.FILES['image']
    result = process_image(image)
    
    # Format response theo yêu cầu bài tập
    detections = []
    for item in result.get('clothing_items', []):
        detections.append({
            'label': item.get('label', ''),
            'score': round(item.get('score', 0), 2),
            'bbox': item.get('bbox', [])
        })
    
    # Ảnh cắt riêng từng item
    cropped_items = result.get('cropped_items', [])
    
    return JsonResponse({
        'success': True,
        'original_url': result.get('original_url', ''),
        'result_url': result.get('result_url', ''),
        'detections': detections,
        'cropped_items': cropped_items  # Thêm ảnh cắt
    })


def process_image(image):
    """
    Xử lý 1 ảnh: detect clothing và tạo ảnh kết quả.
    Dùng chung cho cả view HTML và API.
    """
    # Tạo tên file unique
    ext = os.path.splitext(image.name)[1]
    unique_name = f"{uuid.uuid4()}{ext}"

    # Lưu ảnh gốc
    fs = FileSystemStorage()
    filename = fs.save(f'uploads/{unique_name}', image)
    original_path = fs.path(filename)
    original_url = fs.url(filename)

    # Nhận diện quần áo
    clothing_items = detect_clothing(original_path)

    # Tạo ảnh với đường viền polygon
    result_name = f'results/{unique_name}_polygon.png'
    result_path = os.path.join(settings.MEDIA_ROOT, result_name)
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    
    create_polygon_image(original_path, clothing_items, result_path)
    result_url = settings.MEDIA_URL + result_name

    # Cắt riêng từng item
    cropped_items = crop_clothing_items(original_path, clothing_items, unique_name)

    return {
        'original_url': original_url,
        'result_url': result_url,
        'clothing_items': clothing_items,
        'cropped_items': cropped_items,
    }
