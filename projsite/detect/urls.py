from django.contrib import admin
from django.urls import path

from detect import views

urlpatterns = [
    # add these to configure our home page (default view) and result web page
    path('', views.img_upload, name='image_upload'),
    path('result/', views.result, name='result'),
    path('success', views.success, name = 'success'),

    path('image_upload', views.img_upload, name = 'image_upload'),
    path('process_img', views.process_img, name = 'process_img'),

    path('inference', views.inference, name = 'inference'),
]