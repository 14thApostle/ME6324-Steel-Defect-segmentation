from django.http import HttpResponse, JsonResponse
from django.http.response import Http404
from django.shortcuts import render, redirect

import pdb
import os

import cv2
import time

from albumentations import (Normalize, Compose)
from albumentations.pytorch import ToTensor
import torch
import numpy as np

import pandas as pd
from tqdm import tqdm
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, Dataset
import torch.utils.data as data
import matplotlib.pyplot as plt

from segmentation_models_pytorch import Unet


''''
Note: I am aware that this breaks most (almost all) good practices for a website and Django.
But this is a quick and dirty impl and I am not concerned about the performance right now
:p

TODO:
- Move all inference code to a model
- Async inference
'''


class TestDataset(Dataset):
    '''Dataset for test prediction'''
    def __init__(self, root, df, mean, std):
        self.root = root
        df['ImageId'] = df['ImageId_ClassId'].apply(lambda x: x.split('_')[0])
        self.fnames = df['ImageId'].unique().tolist()
        self.num_samples = len(self.fnames)
        self.transform = Compose(
            [
                Normalize(mean=mean, std=std, p=1),
                ToTensor(),
            ]
        )

    def __getitem__(self, idx):
        fname = self.fnames[idx]
        path = os.path.join(self.root, fname)
        image = cv2.imread(path)
        images = self.transform(image=image)["image"]
        return fname, images

    def __len__(self):
        return self.num_samples
    
def mask_to_contours(image, mask_layer, color):
    """ converts a mask to contours using OpenCV and draws it on the image
    """

    # https://docs.opencv.org/4.1.0/d4/d73/tutorial_py_contours_begin.html
    contours, hierarchy = cv2.findContours(mask_layer, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    image = cv2.drawContours(image, contours, -1, color, 2)
        
    return image




import colorlover as cl

# see: https://plot.ly/ipython-notebooks/color-scales/
colors = cl.scales['4']['qual']['Set3']
labels = np.array(range(1,5))
# combining into a dictionary
palette = dict(zip(labels, np.array(cl.to_numeric(colors))))
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

patches = []
for idx, color in palette.items():
    patch = mpatches.Patch(color=color/255, label='class {}'.format(idx))
    patches.append(patch)

# initialize test dataloader
best_threshold = 0.2
min_size = 700

mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
test_transform = Compose(
    [
        Normalize(mean=mean, std=std, p=1),
        ToTensor(),
    ]
)
# Initialize mode and load trained weights
ckpt_path = "../model_2nd_unet_60epochs.pth"
device = torch.device("cuda")
print("Loading model", ckpt_path)
model = Unet("resnet18", encoder_weights=None, classes=4, activation=None)
model.to(device)
model.eval()
state = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
model.load_state_dict(state["state_dict"])

# our home page view
def home(request):    
    return render(request, 'index.html')

def about(request):    
    return render(request, 'about.html')

def validation(request):    
    return render(request, 'validation.html')

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

def visualise_mask(img_path, mask):
    """ open an image and draws clear masks, so we don't lose sight of the 
        interesting features hiding underneath 
    """
    
    # reading in the image
    image = cv2.imread(img_path)
    image = cv2.resize(image, (1600, 256))
    # going through the 4 layers in the last dimension 
    # of our mask with shape (256, 1600, 4)
    for index in range(mask.shape[-1]):
        
        # indeces are [0, 1, 2, 3], corresponding classes are [1, 2, 3, 4]
        label = index + 1
        
        # add the contours, layer per layer 
        image = mask_to_contours(image, mask[:,:,index], color=palette[label])   
        
    return image


def preprocess_input(img_path, transform):
    image = cv2.imread(img_path)
    image = cv2.resize(image, (1600, 256))
    images = transform(image=image)["image"]
    return images

def post_process(probability, threshold, min_size):
    '''Post processing of each predicted mask, components with lesser number of pixels
    than `min_size` are ignored'''
    mask = cv2.threshold(probability, threshold, 1, cv2.THRESH_BINARY)[1]
    num_component, component = cv2.connectedComponents(mask.astype(np.uint8))
    predictions = np.zeros((256, 1600), np.float32)
    num = 0
    for c in range(1, num_component):
        p = (component == c)
        if p.sum() > min_size:
            predictions[p] = 1
            num += 1
    return predictions, num

# custom method for generating predictions
def inference(request):
    print("At inference")
    # img = cv2.imread('./media/img1.png')
    img_path = './media/img1.png'
    img = preprocess_input(img_path, test_transform)
    img = img.unsqueeze(0)
    img = img.to(device)

    preds = torch.sigmoid(model(img))
    preds = preds.detach().cpu().numpy()
    print("input", img.shape)
    print("output",preds.shape)

    predictions = []
    final_mask = []

    cls_found = []
    for cls, pred in enumerate(preds[0]):
        pred, num = post_process(pred, best_threshold, min_size)
        final_mask.append(pred)
        name = os.path.basename(img_path) + f"_{cls+1}"
        print("for cls", cls, np.unique(pred, return_counts = True))
        predictions.append([name, pred])
        if 1 in list(np.unique(pred)):
            cls_found.append(str(cls+1))

    final_mask = np.array(final_mask)
    final_mask = np.transpose(final_mask, (1,2, 0)).astype(np.uint8)
    print(final_mask.shape)
    print("found classes", cls_found)
    print(np.unique(final_mask, return_counts = True))

    final_img = visualise_mask(img_path=img_path, mask=final_mask)

    print("saving image", final_img.shape)
    plt.figure(figsize = (16, 3))
    plt.legend(handles=patches)
    plt.axis("off")
    plt.imshow(final_img)
    plt.savefig('./detect/static/detect/img2.png', bbox_inches='tight', transparent=True)
    if len(cls_found) == 0:
        txt = "No defect found"
    else:
        txt = "Defect found of class type: {}".format(cls_found)
    print(txt)
    # cv2.imwrite('./detect/static/detect/img2.png', final_img)
    return render(request, 'result.html', {'txt': txt})
    # return redirect('result')        