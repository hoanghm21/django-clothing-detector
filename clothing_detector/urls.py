from django.urls import path
from . import views

app_name = "clothing_detector"

urlpatterns = [
    path("", views.upload_image, name="home"),  # Giao diện web
    path("upload", views.api_upload, name="api_upload"),  # API endpoint POST /upload
]