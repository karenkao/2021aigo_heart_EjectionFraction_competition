# preprocess.py
# author: Ren
# Date: 2021.08.29

import os 
import cv2
import json
import random
import pydicom 
import concurrent.futures
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Callable
from scipy import ndimage
from scipy.ndimage import label
from scipy.ndimage import generate_binary_structure
from scipy.ndimage.interpolation import zoom
from albumentations import Rotate
from albumentations import Transpose
from albumentations import ElasticTransform

def to_binary(arr, threshold, dtype=np.bool):
    arr[arr >= threshold] = 1.
    arr[arr < threshold] = 0.
    return arr.astype(dtype)

def affine_trans(images):
    rows,cols = images.shape[:2]
    #pts1 = np.float32([[50,50],[200,50],[50,200]])
    #pts2 = np.float32([[30,30],[190,40],[60,220]])
    pts1 = np.float32([[40,35],[200,50],[55,200]])
    pts2 = np.float32([[35,30],[195,45],[60,220]])
    M = cv2.getAffineTransform(pts1,pts2)
    res = cv2.warpAffine(images,M,(rows,cols))
    return res

def sift_trans(images):
    rows,cols = images.shape[:2]
#     sign = 1 if random.random() < 0.5 else -1
    x= round(cols*0.015)*-1
    y= round(rows*0.035)*1
    M = np.array([[1,0,x],[0,1,y]], dtype=np.float32)
    res = cv2.warpAffine(images,M,(rows, cols))
    return res

def rotate_trans(images):
    rows,cols = images.shape[:2]
    M = cv2.getRotationMatrix2D((cols//2, rows//2), 15, 0.8)
    res = cv2.warpAffine(images,M,(rows, cols))
    return res

def data_augmentation_2d_array_Non_rigid(arr):
    elastic = ElasticTransform(p=1, 
                           alpha=10, 
                           sigma=1,
                           alpha_affine=10 * 0.03)
    transpose=Transpose(p=1)
    rot_90 = Rotate(limit=(90,90),p=1)
    
    result = [arr]
    for aug in [elastic,transpose,rot_90]:
        augmented = aug(image=arr)
        image_aug = augmented['image']
        result.append(image_aug)
    
    for aug in [affine_trans,sift_trans,rotate_trans]:
        image_aug = aug(arr)
        result.append(image_aug)
    
    img_aug_all = result+[np.flip(i,1) for i in result]
    
    return img_aug_all

def filter_largest_component(mask):
    structure = generate_binary_structure(2,2)
    label_arr, num_features = label(mask, structure)
    sizes = ndimage.sum(mask, label_arr, range(1, num_features + 1))
    max_label = np.where(sizes == sizes.max())[0] + 1
    output = np.asarray(label_arr == max_label, np.bool)
    return output

def filled_mask(mask):
    h, w = mask.shape
    cnts,hierarchy = cv2.findContours(mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    c_max = sorted(cnts, key=cv2.contourArea, reverse=True)[0]
    completed_mask= np.zeros([h, w])
    cv2.drawContours(completed_mask, [c_max],-1,1,-1) 
    return completed_mask.astype(np.bool)

def area_ratio(mask):
    h, w = mask.shape
    ratio = np.round(np.sum(mask)/(h*w), 3)
    return ratio

def global_standarization(image):
    # global standardization of pixels
    mean, std = image.mean(), image.std()
    image = (image - mean) / std
    return image

def limitedEqualize(img, limit = 2):
    clahe = cv2.createCLAHE(clipLimit = limit, tileGridSize = (4,4))
    return clahe.apply(img)

def mask_to_edge(mask):
    h, w = mask.shape
    cnts,hierarchy = cv2.findContours(mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    c_max = sorted(cnts, key=cv2.contourArea, reverse=True)[0]
    completed_edge= np.zeros([h, w])
    cv2.drawContours(completed_edge, [c_max],-1,1,5) 
    return completed_edge.astype(np.bool)

def preprocess_mask(mask):
    largest = filter_largest_component(mask)
    # dtype bool
    filled = filled_mask(largest)
    # resize to (512, 512)
    new_shape = (512, 512)
    zoom_rate = (new_shape[0]/filled.shape[0], new_shape[1]/filled.shape[1])
    filled = zoom(filled.astype(np.int8), zoom = zoom_rate, order = 3).astype(np.bool)
    return filled

def preprocess_image(image):
    # apply clahe
    clahe = limitedEqualize(image)
    # resize to (512, 512)
    new_shape = (512, 512)
    zoom_rate = (new_shape[0]/clahe.shape[0], new_shape[1]/clahe.shape[1])
    clahe = zoom(clahe, zoom = zoom_rate, order = 3)
    # global standarization (dtype float64)
    # std_clahe = global_standarization(clahe)
    # rescale to 0~1, assume that its range is 0~255(not sure)
    std_clahe = clahe/255.
    return std_clahe

def get_dicom_data(dcm_path):
    ds = pydicom.dcmread(dcm_path)
    try:
        spacing = [float(i) for i in ds.ImagerPixelSpacing]
        image = ds.pixel_array
    except AttributeError:
        print('Missing necessary attributes, skip this scan ...')
    return image, spacing

def readnii_simpleitk(nii_path, ifmask = True):
    """
    Read nii or nii.gz file with simpleitk. 
    Args:
        Input: nii path, ifmask or not.
        Output: image array, spacing
    """
    image = sitk.ReadImage(nii_path)
    img_arr = sitk.GetArrayFromImage(image)
    spacing = image.GetSpacing()
    if ifmask:
        dtype = np.bool
    else:
        dtype = np.float32

    img_arr = img_arr.astype(dtype)
    return img_arr, spacing

def find_non_zero_index(mask_arr):
    index_list = []
    for index, mask in enumerate(mask_arr):
        ratio = area_ratio(mask)
        #if np.sum(mask)>0 and ratio>=0.1
        if np.sum(mask)>10:
            index_list.append(index)
            
    #return [min(index_list), max(index_list)]
    return index_list

def sandwich_slice(image_arr, index):
    previous = index-1 if index-1>=0 else 0
    next_one = index+1 if index+1 <=image_arr.shape[0]-1 else index
    # Combine 3 different slices (-1, 0, +1)
    # shape (3, h, w)
    x = np.array([preprocess_image(image_arr[previous, :, :]),
                  preprocess_image(image_arr[index, :, :]),
                  preprocess_image(image_arr[next_one, :, :])])
    # move to channel last
    x = np.moveaxis(x, 0, -1)
    return x

def preprocess_aug_save2npz(args):
    dcm_path = args[0]['image']
    label_path = args[0]['mask']
    save_dir = args[1]
    ifaug = args[2]
    ifsandwich = args[3]
#     if not os.path.isdir(save_dir):
#         os.makedirs(save_dir)    
    try:
        # read volume of image
        image_arr, _ = get_dicom_data(dcm_path)
        # read volume of mask
        mask_arr, _ = readnii_simpleitk(label_path)
        # find non-zero index of mask
        #min_idx, max_idx = find_non_zero_index(mask_arr)
        non_zero_index_list = find_non_zero_index(mask_arr)
        
        #image_arr = image_arr[min_idx:max_idx]
        #mask_arr = mask_arr[min_idx:max_idx]
        
        for index in non_zero_index_list:
            if ifsandwich:
                image = sandwich_slice(image_arr, index)
            else:
                image = image_arr[index]
                image = preprocess_image(image)
            mask = mask_arr[index]
            mask = preprocess_mask(mask)
            if ifaug:
                img_aug_list = data_augmentation_2d_array_Non_rigid(image)
                msk_aug_list = data_augmentation_2d_array_Non_rigid(mask.astype(np.uint8))
                i=0
                for img, msk in zip(img_aug_list, msk_aug_list):
                    save_path = os.path.join(save_dir,
                                             os.path.splitext(os.path.basename(dcm_path))[0]+f"_{index}_{i}"+".npz")
                    #np.savez_compressed(save_path, x=img, y=mask_to_edge(msk.astype(np.bool)))
                    np.savez_compressed(save_path, x=img, y=msk)
                    i+=1
            else:
                save_path = os.path.join(save_dir,
                                         os.path.splitext(os.path.basename(dcm_path))[0]+f"_{index}"+".npz")
                np.savez_compressed(save_path, x=image, y=mask)
        return f"success: {dcm_path}"
    except Exception as e:
        print(e)
        return f"error: {dcm_path}"

def concurrent_multi_process(list_:list, function_:Callable, *para):
    """
    Implement multi-process to speed up process time.
    Args:
        Input: list, function
        output: list of function's output

    """
    args = ((element, *para) for element in list_)
    with concurrent.futures.ProcessPoolExecutor() as executor:
        result_list = list(tqdm(executor.map(function_, args), total = len(list_)))
        
    return result_list

if __name__ == "__main__":
    # parameters
    seed  = 10
    #ratio = [0.8, 0.1, 0.1]
    ratio = [0.8, 0.2]
    save_root = "../mannual_annotation_dataset/heart_npz/fine_tune_3stage_sandwich_rescale255"
    train_save_dir = os.path.join(save_root, "train")
    val_save_dir = os.path.join(save_root, "val")
    file_root = "../mannual_annotation_dataset/fine_tune"
    ifsandwich = True
    #test_save_dir = os.path.join(save_root, "test")
    # check dir exists
    if not os.path.isdir(save_root):
        os.makedirs(save_root)
    if not os.path.isdir(train_save_dir):
        os.makedirs(train_save_dir)
    if not os.path.isdir(val_save_dir):
        os.makedirs(val_save_dir)
    # start 
    dcm_nii_list = []
    caseid_list = os.listdir(file_root)
    for caseid in caseid_list:
        dcm_nii = {}
        for root, dir_, files in os.walk(os.path.join(file_root, caseid)):
            for file in files:
                if file.endswith(".dcm"):
                    dcm_nii['image'] = os.path.join(root, file)
                if file.endswith(".nii.gz"):
                    dcm_nii['mask'] = os.path.join(root, file)
        dcm_nii_list.append(dcm_nii)
    random.Random(seed).shuffle(dcm_nii_list)
    cut_index_start = int(len(dcm_nii_list)*ratio[0])
    #cut_index_end = int(len(file_path_list)*ratio[1])
    train_list = dcm_nii_list[:cut_index_start]
    val_list = dcm_nii_list[cut_index_start:]
    #val_list = file_path_list[cut_index_start:cut_index_start+cut_index_end]
    #test_list = file_path_list[cut_index_start+cut_index_end:]
    with open(os.path.join(save_root, "train_val_split.json"), "w") as f:
        json.dump({"train": train_list, "val": val_list}, f)
    print("total", len(caseid_list))
    print("train", len(train_list))
    print("val", len(val_list))
    #print("test", len(test_list))
    train_result = concurrent_multi_process(train_list, 
                                            preprocess_aug_save2npz, 
                                            train_save_dir, 
                                            True,
                                            ifsandwich)
    val_result = concurrent_multi_process(val_list, 
                                          preprocess_aug_save2npz, 
                                          val_save_dir, 
                                          False,
                                          ifsandwich)
    #test_result = concurrent_multi_process(test_list, preprocess_aug_save2npz, test_save_dir, label, False)
    with open(os.path.join(save_root, "preprocess_result.json"), "w") as f:
        json.dump({"train": train_result, "val": val_result}, f)
    print("finish all")
