import os
import cv2
from matplotlib import pyplot as plt
import numpy as np
import operator

def preProcess(img):
    gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    equ = cv2.equalizeHist(gray)
        
    img_sobel_x = cv2.Sobel(equ, cv2.CV_64F, 1, 0, ksize=3) 
    img_sobel_x = cv2.convertScaleAbs(img_sobel_x) 
    
    img_sobel_y = cv2.Sobel(equ, cv2.CV_64F, 0, 1, ksize=3) 
    img_sobel_y = cv2.convertScaleAbs(img_sobel_y) 
    
    img_sobel = cv2.addWeighted(img_sobel_x, 1, img_sobel_y, 1, 0)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2,2))
    dilation = cv2.dilate(img_sobel, kernel, iterations=1)
    final = cv2.cvtColor(dilation,cv2.COLOR_GRAY2RGB)
    return final


def detectDeposit(img):
    height,width,_= img.shape
    y1=(height//3)*2
    y2=height
    x1=width//3
    x2=x1*2
        
    roi=img[y1:y2,x1:x2]
    roi_mat=np.array(roi)
    average=np.average(roi_mat)
    # Accuracy is not certain since it is based on around 600 data.
    if average>200:
        return 0.78
    elif average>180:
        return 0.61
    elif average>150:
        return 0.55
    else:
        return 0
    
def contouring(img):
    h,w = img.shape[0], img.shape[1]
    # filter only white color
    img_cvt = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    lower = np.uint8([0,200,0])
    upper = np.uint8([255,255,255])
    white_mask = cv2.inRange(img_cvt, lower, upper)
    # Gaussian Blur for decreasing noise
    blur = cv2.GaussianBlur(white_mask, (7,7), 0)
    # Find contours
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9,9))
    thresh = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,11)
    contours,hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    max_list=[]
    area_list=[]
    for cnt in contours: # find the maximum of area among contours
        area = cv2.contourArea(cnt)
        area_list.append(area)
    max_list = area_list.copy()
    index,max_area = max(enumerate(area_list), key=operator.itemgetter(1))

    
    roi = img[4:h-4, 4:w-4] #remove the wrong contour(edge of the full image)
    # filter only red color
    roi_cvt = cv2.cvtColor(roi, cv2.COLOR_BGR2HLS)
    lower = np.uint8([0,50,50])
    upper = np.uint8([10,255,255])
    red_mask = cv2.inRange(roi_cvt, lower,upper)
    # Find contours
    thresh = cv2.adaptiveThreshold(red_mask,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,11)
    contours2,hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    if len(contours) != 0:
        cv2.drawContours(red_mask, contours[index], -1, (255,0,255), 3)
    
    return red_mask

    
if  __name__ == "__main__":
   
    images=os.listdir("test_image/")
    for image in images:
        img_dir = "test_image/"+image
        img = cv2.imread(img_dir)
        testImage=preProcess(img)
        dep_result=detectDeposit(testImage)
        
        if(dep_result != 0):
            os.rename(img_dir, "result/deposit/"+image+".jpg")
        else:
            os.rename(img_dir, "result/non_deposit/"+image+".jpg")
   
    
    
