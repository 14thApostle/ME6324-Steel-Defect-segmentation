from django.http import HttpResponse, JsonResponse
from django.http.response import Http404
from django.shortcuts import render, redirect

import cv2
import time

# our home page view
def home(request):    
    return render(request, 'index.html')


def img_upload(request):
    # print("Hereeee form get")
    return render(request, 'image_upload_form.html')

def process_img(request):
    if request.method == 'POST':
        handle_uploaded_file(request.FILES['defect_img'])
        return redirect('success')
        # return redirect('inference')
    else:
        raise Http404('Invalid request type {}'.format(request.method))

def handle_uploaded_file(f):
    with open('./media/img1.png', 'wb+') as destination:
        for chunk in f.chunks():
            destination.write(chunk)

def success(request):
    responseData = {
        'success': True,
    }
    return JsonResponse(responseData)

def result(request):
    return render(request, 'result.html')

# custom method for generating predictions
def inference(request):
    print("At inference")
    img = cv2.imread('./media/img1.png')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (416, 416))
    blurImg = cv2.blur(img,(10,10)) # blur the image
    cv2.imwrite('img3.png', blurImg)
    cv2.imwrite('./detect/static/detect/img2.png', blurImg)
    return render(request, 'result.html')
    # return redirect('result')        