# inference.py
# Author: Ren
# Date: 2021.08.31
#channel last
import os
import sys
import cv2
import imutils
import argparse
import keras
import time
import numpy as np

from keras.models import load_model
from model_architecture_vgg16_encoder_unet import unet
from model_architecture_vgg16_encoder_unet import dice_coef_loss
from model_architecture_vgg16_encoder_unet import dice_coef
from model_architecture_vgg16_encoder_unet import focal_loss
from model_architecture_vgg16_encoder_unet import Combinations_loss
from preprocess_aug_savenpz import preprocess_image
from preprocess_aug_savenpz import get_dicom_data
from preprocess_aug_savenpz import readnii_simpleitk
from preprocess_aug_savenpz import sandwich_slice
from preprocess_aug_savenpz import to_binary

def seg_model(h5):
    #model= load_model(h5,custom_objects={'dice_coef_loss': dice_coef_loss,'dice_coef':dice_coef})
    model= load_model(h5,custom_objects={'focal_loss': focal_loss,'dice_coef':dice_coef})
    print("load model success")
    return model

def seg_inference(model,series_path,threshold=0.5, ifsandwich=True ,ifnii=False):
    extension = os.path.splitext(series_path)[-1]
    if extension == ".nii" or extension == ".nii.gz":
    #if ifnii:
        image_arr, spacing = readnii_simpleitk(series_path, ifmask=False)
    else:
        image_arr, spacing = get_dicom_data(series_path)
    
    y_pred_list=[]
    x_list=[]
    for index, image in enumerate(image_arr):
        #h, w = image.shape
        x_list.append(image)
        if ifsandwich:
            # shape (512, 512, 3)
            image = sandwich_slice(image_arr, index)
        else:
            # shape (512, 512, 1)
            image = preprocess_image(image)
            image = np.expand_dims(image, axis = 2)
        h, w, _ = image.shape
        image = np.expand_dims(image, axis=0)
        pred = model.predict(image)
        
        pred = pred.reshape((h, w))
        y_pred_list.append(to_binary(pred, threshold))
        
    y_pred = np.array(y_pred_list)
    x = np.array(x_list)
    return y_pred, x

def seg_inference_batch(model,series_path,threshold=0.5, ifsandwich=True ,ifnii=False, batch_size=32):
    start = time.time()
    extension = os.path.splitext(series_path)[-1]
    if extension == ".nii" or extension == ".nii.gz":
    #if ifnii:
        image_arr, spacing = readnii_simpleitk(series_path, ifmask=False)
    else:
        image_arr, spacing = get_dicom_data(series_path)
    print("read image", time.time()-start)
    
    start_b = time.time()
    ori_x_list=[]
    x_list = []
    for index, image in enumerate(image_arr):
        #h, w = image.shape
        ori_x_list.append(image)
        if ifsandwich:
            # shape (512, 512, 3)
            image = sandwich_slice(image_arr, index)
        else:
            # shape (512, 512, 1)
            image = preprocess_image(image)
            image = np.expand_dims(image, axis = 2)
        h, w, _ = image.shape
        x_list.append(image)
    image_arr = np.array(x_list)
    print(image_arr.shape)
    print("preprocess", time.time()-start_b)
    start_c = time.time()
    y_pred_list=[]
    for i in range(0, image_arr.shape[0], batch_size):
        images = image_arr[i:i+batch_size]
        images = np.reshape(images, images.shape)
        pred = model.predict(images)
        
        pred = np.squeeze(pred, 3)
        for y_pred in pred:
            y_pred_list.append(to_binary(y_pred, threshold))
    print("pred", time.time()-start_c)
    y_pred = np.array(y_pred_list)
    x = np.array(ori_x_list)
    return y_pred, x
    
smooth = 1.
def dice_coef_np(y_true, y_pred):
    y_true_f = np.ndarray.flatten(y_true)
    y_pred_f = np.ndarray.flatten(y_pred)
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (np.sum(y_true_f*y_true_f) + np.sum(y_pred_f*y_pred_f) + smooth)

def convexhull(img, mask):
    convex_img = img.copy()
    mask = np.array(mask * 255, dtype=np.uint8)
    convex_mask = np.zeros(mask.shape, dtype=np.uint8)
    contours = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    if contours:
        cnt = max(contours, key=cv2.contourArea)
        #for cnt in contours: 
        hull = cv2.convexHull(cnt)
        length = len(hull)
        #print(length)
        # 如果凸包点集中的点个数大于5
        if length > 3:
            # 绘制图像凸包的轮廓
            for i in range(length):
                cv2.line(convex_img, tuple(hull[i][0]), tuple(hull[(i+1)%length][0]), (0,0,255), 2)
                cv2.drawContours(convex_mask, [hull], -1, (255,255,255), -1)
    return convex_img, convex_mask

if __name__=='__main__':        
    h5 = sys.argv[1]
    series_path =sys.argv[2]
    threshold = sys.argv[3]
    inference(h5,series_path,threshold)
    
    
    
    
    
    
    