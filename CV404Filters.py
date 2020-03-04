# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 23:27:01 2020

@author: شيماء
"""

from PyQt5 import QtWidgets,QtGui , QtCore ,Qt
from PyQt5.QtWidgets import   QFileDialog  ,QWidget,QApplication
from PyQt5.QtGui import QPixmap
from MainWindow import Ui_MainWindow
from PIL import Image
from PIL.ImageQt import ImageQt
import sys
from os import listdir
from os.path import isfile , join
import numpy as np
import qimage2ndarray
from scipy import ndimage
#import array2qimage 
import math
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


class Filters(QtWidgets.QMainWindow):
    def __init__(self):
        super(Filters, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.ui.pushButton_filters_load.clicked.connect(self.button_clicked)
        self.ui.comboBox.currentIndexChanged.connect(self.add_noise)
        self.ui.comboBox_2.currentIndexChanged.connect(self.show_filters) 
     
    def button_clicked(self):  
        self.fileName, self.filter = QFileDialog.getOpenFileName(self, "Title"," " , "Filter -- img file (*.jpg *.PNG);;img file (*.PNG)")
        if self.fileName:
            
            pixmap = QPixmap(self.fileName)
            self.pixmap = pixmap.scaled(512, 512, QtCore.Qt.KeepAspectRatio, QtCore.Qt.FastTransformation) 
            self.color_img =mpimg.imread(self.fileName)
            self.gray_img =self.rgb2gray(self.color_img)
            #self.gray_img =self.rgb2gray(self.color_img)
             
            #plt.imshow(self.gray_img, cmap=plt.get_cmap('gray'), vmin=0, vmax=1)
            self.Display_image() 
            
    def rgb2gray(self,rgb_image):
        return np.dot(rgb_image[...,:3], [0.299, 0.587, 0.114])

    def Display_image(self):
        if self.filter=="House.jpg" or self.filter=="Pyramids2.jpg" or self.filter=="some-pigeon.jpg":
             self.input_iamge=np.array(self.gray_img)
        else:
             self.input_iamge=np.array(self.gray_img*200)
        #print (input_iamge)
        self.input_iamge=qimage2ndarray.array2qimage(self.input_iamge)
        self.input_iamge=QPixmap(self.input_iamge)
        self.ui.label_filters_input.setPixmap(self.input_iamge)
        self.ui.label_filters_input.show()
#        plt.figure(figsize=(10,20)) #divid plot ,each image take 10x20 size
#        plt.imshow( self.filter )
#        plt.imshow(self.gray_img, cmap=plt.get_cmap('gray'), vmin=0, vmax=1)
        
    def corr(self,mask):
        row,col=self.gray_img.shape
        m,n=mask.shape
        new=np.zeros((row+m-1,col+n-1))
        n=n//2
        m=m//2
        filtered=np.zeros(self.gray_img.shape)
        new[m:new.shape[0]-m,n:new.shape[1]-n]=self.gray_img
        for i in range (m,new.shape[0]-m):
            for j in range (n,new.shape[1]-n):
                temp=new[i-m:i+m+1,j-m:j+m+1]
                result=temp*mask
                filtered[i-m,j-n]=result.sum()      
        return filtered

    def gaussian(self,m,n,sigma):
        gaussian=np.zeros((m,n))
        m=m//2
        n=n//2
        for x in range (-m,m+1):
            for y in range (-n,n+1):
                x1=sigma*math.sqrt(2*np.pi)
                x2=np.exp(-(x**2+y**2)/(2*sigma**2))
                gaussian[x+m,y+n]=(1/x1)*x2  
        return gaussian

    def gaussian_filter(self,m,n,sigma):
        g=self.gaussian(m,n,sigma)
        n=self.corr(g)
        return n
    
    def mean(self,k):
        #meanFilter=[]
        n=k*k
        print(n)
        meanFilter=(np.ones((k,k)))*(1/n)
        #print(meanFilter)
        filt=self.corr(meanFilter)
        #filt=ndimage.convolve(self.gray_img, meanFilter, mode='constant', cval=0.0)
        print("9898")
        return filt
    
    def median_filter(self,mask):
        m,n=self.gray_img.shape
        median = np.zeros((m,n))
        temp = []
        mask_center = mask // 2  # to get the center value
        
        #print(m)
        #print(n)
    
        for i in range(m):
            for j in range(n): #(i,j)for loop for image
                for u in range(mask):
                    for v in range(mask):#(u,v)for loop for image 
                       if (i + u - mask_center < 0) or (i + u - mask_center > m - 1):
                            temp.append(0)
                       elif (j + u - mask_center < 0) or (j + mask_center > n - 1):
                            temp.append(0)
                       else:                     
                         temp.append(self.gray_img[i + u - mask_center][j + v - mask_center])
    
                temp.sort()
                median[i][j] = temp[len(temp) // 2]
                temp = []
        return median
    
    
    
    def gaussian_noise( self,mu, sigma, im_size ):
        randGaussian=np.random.normal( mu, sigma, im_size) #np.random.normal Gaussian noise
        return randGaussian
    
    def im_gaussian_noise(self,mu, sigma):
        g_noise= self.gaussian_noise(mu,sigma, self.gray_img.shape)
        img_w_g_noise = self.gray_img + g_noise
        return img_w_g_noise
    
            
    def Random_Uniform(self,percent):
        img_noisy=np.zeros(self.gray_img.shape)
        uniform = np.random.random(self.gray_img.shape) 
        cleanPixels_ind=uniform > percent
        noise = (uniform <= (percent)); 
        img_noisy[cleanPixels_ind]=self.gray_img[cleanPixels_ind]
        img_noisy[noise] = 0.7
        return img_noisy           
            
    
    def salt_pepper_noise(self,percent):
        img_noisy=np.zeros(self.gray_img.shape)
        salt_pepper = np.random.random(self.gray_img.shape) # Uniform distribution
        cleanPixels_ind=salt_pepper > percent
        #NoisePixels_ind=salt_pepper <= percent
        pepper = (salt_pepper <= (0.5* percent)); # pepper < half percent
        
        salt = ((salt_pepper <= percent) & (salt_pepper > 0.5* percent)); 
        img_noisy[cleanPixels_ind]=self.gray_img[cleanPixels_ind]
        img_noisy[pepper] = 0
        img_noisy[salt] = 1
        return img_noisy
  
    def show_filters(self): 
        self.filters = str(self.ui.comboBox_2.currentText())
        
        if self.filters=="Gaussian":
            #self.input_iamge=array2qimage.qimage2ndarray(self.input_iamge)
            self.filter_img =self.gaussian_filter(5,5,2)
            self.filter=np.array(self.filter_img*50)
            self.input_iamge=qimage2ndarray.array2qimage(self.filter)
            self.output_iamge=QPixmap(self.input_iamge)
            self.ui.label_filters_output.setPixmap(self.output_iamge)
            self.ui.label_filters_output.show()
        
        elif self.filters=="Mean":    
                self.filter_img =self.mean(5)
                self.filter=np.array(self.filter_img*200)
                self.input_iamge=qimage2ndarray.array2qimage(self.filter)
                self.output_iamge=QPixmap(self.input_iamge)
                self.ui.label_filters_output.setPixmap(self.output_iamge)
                self.ui.label_filters_output.show() 
                 
#       
        elif self.filters=="Median":    
            self.filter_img =self.median_filter(5)
            self.filter=np.array(self.filter_img*200)
            self.input_iamge=qimage2ndarray.array2qimage(self.filter)
            self.output_iamge=QPixmap(self.input_iamge)
            self.ui.label_filters_output.setPixmap(self.output_iamge)
            self.ui.label_filters_output.show()  
        
        else:
            print("2")
    #def add_noise(self):
    def add_noise(self): 
            self.filters = str(self.ui.comboBox.currentText())
            
            if self.filters=="Gaussian":
                #self.input_iamge=array2qimage.qimage2ndarray(self.input_iamge)
                self.filter_img =self.im_gaussian_noise(0,0.3)
                self.filter=np.array(self.filter_img*200)
                self.input_iamge=qimage2ndarray.array2qimage(self.filter)
                self.output_iamge=QPixmap(self.input_iamge)
                self.ui.label_filters_output.setPixmap(self.output_iamge)
                self.ui.label_filters_output.show()
            
            elif self.filters=="Uniform":    
                self.filter_img =self. Random_Uniform(0.3)
                self.filter=np.array(self.filter_img*300)
                self.input_iamge=qimage2ndarray.array2qimage(self.filter)
                self.output_iamge=QPixmap(self.input_iamge)
                self.ui.label_filters_output.setPixmap(self.output_iamge)
                self.ui.label_filters_output.show() 
                print("1212")
    #       
            elif self.filters=="Salt-papper":    
                self.filter_img =self.salt_pepper_noise(0.3)
                self.filter=np.array(self.filter_img*200)
                self.input_iamge=qimage2ndarray.array2qimage(self.filter)
                self.output_iamge=QPixmap(self.input_iamge)
                self.ui.label_filters_output.setPixmap(self.output_iamge)
                self.ui.label_filters_output.show()  
            
                





def main():
    app = QtWidgets.QApplication(sys.argv)
    application = Filters()
    application.show()
    sys.exit(app.exec_())
    
   


if __name__ == "__main__":
    main()        
            
        