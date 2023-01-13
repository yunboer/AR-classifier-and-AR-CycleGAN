import tifffile
import cv2
import os
import numpy as np



def process(src:str, save_root:str):
    '''
        process all the cams in src
        save the result in save_root
        name won't be changed.
    '''
    cam_names = os.listdir(src)
    for cam_name in cam_names:
        cam = tifffile.imread(os.path.join(src,cam_name))
        processed_cam = pack(cam, iter = 3)
        cv2.imwrite(os.path.join(save_root,cam_name),processed_cam)

        
def seq(image):
    k = np.ones((16,16),dtype=np.float32)/256.0
    image = cv2.filter2D(image,-1,kernel=k)
    cv2.normalize(image,image,0,1,cv2.NORM_MINMAX)
    image = cv2.medianBlur(image, 5)
    cv2.threshold(image,0.2,1,cv2.THRESH_TOZERO,image)
    
    return image

def pack(image,iter):
    cv2.threshold(image,0.1,1,cv2.THRESH_TOZERO,image)
    image = cv2.medianBlur(image, 5)
    image = cv2.resize(image,(256,256))
    for i in range(iter):
        image = seq(image)
    image = cv2.resize(image,(28,28))
    
    return image


process(
    src = r"D:\BaiduDownload\NCT-CRC-HE-100k-CMP\NCT-CRC-HE-100K-detect-1.0-CAM\cam",
    save_root = r"D:\BaiduDownload\NCT-CRC-HE-100k-CMP\NCT-CRC-HE-100K-CAM"
)