# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 15:23:12 2020

@author: Practica
"""

from PIL import Image 
import glob
#from LocalBinaryPatterns import LocalBinaryPatterns
import matplotlib.pyplot as plt
import os
from sklearn.svm import SVC
from sklearn import metrics
from skimage import data
from skimage.color import rgb2gray
import cv2
from getLBPimage import getLBPimage
import numpy as np

image_list_sanatoase = []

for filename in glob.glob(r'C:\Users\Practica\Desktop\Baza de date organizata- funze\frunze de vita de vie\Antrenare\healthy/*.jpg'):
    im = cv2.imread(filename)
    image_list_sanatoase.append(im)
    
image_list_infectate = []

for filename in glob.glob(r'C:\Users\Practica\Desktop\Baza de date organizata- funze\frunze de vita de vie\Antrenare\infected/*.jpg'):
    im = cv2.imread(filename)
    image_list_infectate.append(im)

nr_imagini_infectate = len(image_list_infectate)
print(nr_imagini_infectate)
desc = LocalBinaryPatterns(8, 1)

data = []

cnt = 0
for i in range (0, nr_imagini_sanatoase):
    imgLBP = getLBPimage(image_list_sanatoase[i])
    hist = imgLBP.flatten()
    data.append(hist)
    cnt = cnt + 1
    print(cnt)
    
print('VERIFICARE1')
cnt = 0 
for i in range (0, nr_imagini_infectate):
    imgLBP = getLBPimage(image_list_infectate[i])
    hist = imgLBP.flatten()
    data.append(hist)
    cnt = cnt + 1
    print(cnt)
    
print('VERIFICARE1')    
X = data
Y = []

for i in range (0, nr_imagini_sanatoase):
    Y.append(0)
for i in range (0, nr_imagini_infectate):
    Y.append(1)
    
clf = SVC(kernel='linear')
clf.fit(X, Y)

print('*****************************************************************')


image_list_testare = []
for filename in glob.glob(r'C:\Users\Practica\Desktop\Baza de date organizata- funze\frunze de vita de vie\Testare\healthy/*.jpg'):
    im = cv2.imread(filename)
    image_list_testare.append(im)

nr_imagini_testare_sanatoase = len(image_list_testare)
    
for filename in glob.glob(r'C:\Users\Practica\Desktop\Baza de date organizata- funze\frunze de vita de vie\Testare\infected/*.jpg'):
    im = cv2.imread(filename)
    image_list_testare.append(im)
    
nr_imagini_testare = len(image_list_testare)
nr_imagini_testare_infectate = nr_imagini_testare - nr_imagini_testare_sanatoase

X_testare = []
cnt = 0
for i in range (0, nr_imagini_testare):
    imgLBP = getLBPimage(image_list_testare[i])
    hist = imgLBP.flatten()
    X_testare.append(hist)
    cnt = cnt + 1
    print(cnt)

Y_real =[]
cnt = 0
for i in range (0, nr_imagini_testare_sanatoase):
    Y_real.append(0)
for i in range (nr_imagini_testare_sanatoase, nr_imagini_testare):
    Y_real.append(1)

Y_prezis = clf.predict(X_testare)
print(Y_prezis)
print("Acuratete(%):", metrics.accuracy_score(Y_real, Y_prezis)*100)


