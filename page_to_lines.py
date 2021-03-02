# -*- coding: utf-8 -*-
"""
Created on Mon May 18 23:08:06 2020

@author: JOEL
"""
import cv2
import numpy as np
def crop(imgpath):

 ## (1) read
 img = cv2.imread("upload/filen.png") 
 gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
 a,b = img.shape[:2]
 ## (2) threshold
 th, threshed = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)
 

 ## (3) minAreaRect on the nozeros
 pts = cv2.findNonZero(threshed)
 ret = cv2.minAreaRect(pts)

 (cx,cy), (w,h), ang = ret
 if w<h:
    w,h = h,w
    ang += 90

 ## (4) Find rotated matrix, do rotation
 M = cv2.getRotationMatrix2D((cx,cy), ang, 1.0)
 rotated = cv2.warpAffine(threshed, M, (img.shape[1], img.shape[0]))

 ## (5) find and draw the upper and lower boundary of each lines
 hist = cv2.reduce(rotated,1, cv2.REDUCE_AVG).reshape(-1)

 th = 2
 H,W = img.shape[:2]
 uppers = [y for y in range(H-1) if hist[y]<=th and hist[y+1]>th]
 lowers = [y for y in range(H-1) if hist[y]>th and hist[y+1]<=th]

 rotated = cv2.cvtColor(rotated, cv2.COLOR_GRAY2BGR)
 '''for y in uppers:
    cv2.line(rotated, (0,y), (W, y), (255,0,0), 1)

 for y in lowers:
    cv2.line(rotated, (0,y), (W, y), (0,255,0), 1)'''
 rotated = ~rotated
 cropped = []
 for i in range(len(uppers)):
    #print(type(w))
    cropped.append(rotated[uppers[i]:lowers[i], 0:b ])
 #print(type(uppers))    

 #th, rotated = cv2.threshold(rotated, 127, 255, cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)
 for i in range(len(cropped)):
    cv2.imwrite("temp/{}.png".format(str(i)),cropped[i])
 '''cv2.imwrite("result.png", rotated)
 window_name= 'image'
 image = cv2.imread("result.png")
 cv2.imshow(window_name,image)
 cv2.waitKey(0)'''
 return len(cropped)

