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
from scipy.ndimage.filters import convolve
import cv2

#
#
#imgs_final = []
#img_smoothed = None
#gradientMat = None
#thetaMat = None
#nonMaxImg = None
#thresholdImg = None
#weak_pixel = 75
#strong_pixel = 255
#sigma = 1
#kernel_size = 5
#lowThreshold = 0.05
#highThreshold = 0.15


class Filters(QtWidgets.QMainWindow):
    def __init__(self):
        super(Filters, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.ui.pushButton_filters_load.clicked.connect(self.button_clicked)
        self.ui.comboBox.currentIndexChanged.connect(self.add_noise)
        self.ui.comboBox_2.currentIndexChanged.connect(self.show_filters)
        self.imgs_final = []
        self.img_smoothed = None
        self.gradientMat = None
        self.thetaMat = None
        self.nonMaxImg = None
        self.thresholdImg = None
        self.weak_pixel = 75
        self.strong_pixel = 255
        self.sigma = 1
        self.kernel_size = 5
        self.lowThreshold = 0.05
        self.highThreshold = 0.15

     
    def button_clicked(self):  
        self.fileName, self.filter = QFileDialog.getOpenFileName(self, "Title"," " , "Filter -- img file (*.jpg *.PNG);;img file (*.PNG)")
        if self.fileName:
            
            pixmap = QPixmap(self.fileName)
            self.pixmap = pixmap.scaled(512, 512, QtCore.Qt.KeepAspectRatio, QtCore.Qt.FastTransformation) 
            self.color_img =mpimg.imread(self.fileName)
            self.gray_img =self.rgb2gray(self.color_img)
            self.ui.lineEdit.setText(""+('image')+"")
            self.ui.lineEdit_2.setText(""+str(self.gray_img.shape[0])+""+str('x')+""+str(self.gray_img.shape[1])+"")
            #print(self.fileName[66:100])
            #for i in range (self.fileName[6]):
            self.Display_image() 
            
    def rgb2gray(self,rgb):
        #return np.dot(rgb_image[...,:3], [0.299, 0.587, 0.114])
          r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
          gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
          return gray
    def Display_image(self):
        if self.fileName[66:100]=="House.jpg" or self.fileName[66:100]=="Pyramids2.jpg" or self.fileName[66:100]=="some-pigeon.jpg":
             self.input_iamge=np.array(self.gray_img)
        else:
             self.input_iamge=np.array(self.gray_img*200)
        #print (input_iamge)
        self.input_iamge=qimage2ndarray.array2qimage(self.input_iamge)
        self.input_iamge=QPixmap(self.input_iamge)
        self.ui.label_filters_input.setPixmap(self.input_iamge)
        self.ui.label_filters_input.show()

        
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
        #print("9898")
        return filt
    
    def median_filter(self,mask):
        m,n=self.gray_img.shape
        median = np.zeros((m,n))
        temp = []
        mask_center = mask // 2  # to get the center value

    
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

    def prewitt(self):
        n,m = np.shape(self.gray_img)
        Gx= np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
        Gy= np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
         
        filt= np.zeros(shape=(n, m))
        
        for i in range(n - 2):
            for j in range(m - 2):
                gx = np.sum(np.multiply(Gx, self.gray_img[i:i + 3, j:j + 3])) 
                gy = np.sum(np.multiply(Gy, self.gray_img[i:i + 3, j:j + 3])) 
                filt[i + 1, j + 1] = np.sqrt(gx ** 2 + gy ** 2)
        
        return filt



    def robert(self):
            n,m= np.shape(self.gray_img) 
            Gx = np.array([[0, 0, 0], [0, 1, 0], [0, 0, -1]])
            Gy = np.array([[0, 0, 0], [0, 0, 1], [0, -1, 0]])
            filt= np.zeros(shape=(n, m)) 
            
            for i in range(n - 2):
                for j in range(m - 2):
                    gx = np.sum(np.multiply(Gx, self.gray_img[i:i + 3, j:j + 3]))  
                    gy = np.sum(np.multiply(Gy, self.gray_img[i:i + 3, j:j + 3]))  
                    filt[i + 1, j + 1] = np.sqrt(gx ** 2 + gy ** 2)  
            
            return  filt


    def sobel(self):
            n,m= np.shape(self.gray_img)
            Gx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
            Gy = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
              
            filt= np.zeros(shape=(n, m))  
            
            for i in range(n - 2):
                for j in range(m - 2):
                    gx = np.sum(np.multiply(Gx, self.gray_img[i:i + 3, j:j + 3]))  
                    gy = np.sum(np.multiply(Gy, self.gray_img[i:i + 3, j:j + 3]))  
                    filt[i + 1, j + 1] = np.sqrt(gx ** 2 + gy ** 2)  
            
            return  filt
    
    
    
    def sobel_filter_for_canny(self,img):
        Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
        Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)

        Ix = ndimage.filters.convolve(img, Kx)
        Iy = ndimage.filters.convolve(img, Ky)

        G = np.hypot(Ix, Iy)
        G = G / G.max() * 255
        theta = np.arctan2(Iy, Ix)
        return (G, theta)
    
    def non_max_suppression(self,img, D):
        M, N = img.shape
        Z = np.zeros((M,N), dtype=np.int32)
        angle = D * 180. / np.pi
        angle[angle < 0] += 180


        for i in range(1,M-1):
            for j in range(1,N-1):
                try:
                    q = 255
                    r = 255

                   #angle 0
                    if (0 <= angle[i,j] < 22.5) or (157.5 <= angle[i,j] <= 180):
                        q = img[i, j+1]
                        r = img[i, j-1]
                    #angle 45
                    elif (22.5 <= angle[i,j] < 67.5):
                        q = img[i+1, j-1]
                        r = img[i-1, j+1]
                    #angle 90
                    elif (67.5 <= angle[i,j] < 112.5):
                        q = img[i+1, j]
                        r = img[i-1, j]
                    #angle 135
                    elif (112.5 <= angle[i,j] < 157.5):
                        q = img[i-1, j-1]
                        r = img[i+1, j+1]

                    if (img[i,j] >= q) and (img[i,j] >= r):
                        Z[i,j] = img[i,j]
                    else:
                        Z[i,j] = 0


                except IndexError as e:
                    pass

        return Z
    
    def threshold(self,img):
#        lowThreshold = 0.05
#        self.highThreshold = 0.1
        LhighThreshold = img.max() * self.highThreshold;
        LlowThreshold = self.highThreshold * self.lowThreshold;

        M, N = img.shape
        res = np.zeros((M,N), dtype=np.int32)

        weak = np.int32(self.weak_pixel)
        strong = np.int32(self.strong_pixel)

        strong_i, strong_j = np.where(img >= LhighThreshold)
        zeros_i, zeros_j = np.where(img < LlowThreshold)

        weak_i, weak_j = np.where((img <= LhighThreshold) & (img >= LlowThreshold))

        res[strong_i, strong_j] = strong
        res[weak_i, weak_j] = weak

        return (res)

    def hysteresis(self,img):

        M, N = img.shape
        weak = self.weak_pixel
        strong = self.strong_pixel

        for i in range(1, M-1):
            for j in range(1, N-1):
                if (img[i,j] == weak):
                    try:
                        if ((img[i+1, j-1] == strong) or (img[i+1, j] == strong) or (img[i+1, j+1] == strong)
                            or (img[i, j-1] == strong) or (img[i, j+1] == strong)
                            or (img[i-1, j-1] == strong) or (img[i-1, j] == strong) or (img[i-1, j+1] == strong)):
                            img[i, j] = strong
                        else:
                            img[i, j] = 0
                    except IndexError as e:
                        pass

        return img


    
    

###########################canny
    def canny_filter(self):
              #self.gray_img
#              
            #self.gray_img =cv2.imread(self.fileName,0)
            canny_img_final = []
            input_size=(self.ui.lineEdit_3.text())
            #plt.imshow(self.gray_img)
            guass=self.gaussian(int(input_size),int(input_size),1)
            self.img_smoothed = convolve(self.gray_img,guass)
            self.gradientMat, self.thetaMat =self.sobel_filter_for_canny(self.img_smoothed)
            self.nonMaxImg =self.non_max_suppression(self.gradientMat, self.thetaMat)
            self.thresholdImg =self.threshold(self.nonMaxImg)
            img_final = self.hysteresis(self.thresholdImg)
            canny_img_final.append(img_final)
            print(canny_img_final)
            self.visualize(canny_img_final, 'gray')

    def visualize(self,imgs, format=None, gray=False):
            #plt.figure(figsize=(20, 40))
            for i, img in enumerate(imgs):
                if img.shape[0] == 3:
                    print("rrrrrrrrrrrrr")
                    img = img.transpose(1,2,0)
                img=Image.fromarray(np.uint8(img))    
                img.save("canny_edges.jpg")    
#                plt_idx = i+1
#                plt.subplot(2, 2, plt_idx)
#                plt.imshow(img, format)
#            plt.show()
#            
        
     
######################################### add noise#################################    
    
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
            input_size=(self.ui.lineEdit_3.text())
            self.filter_img =self.gaussian_filter(int(input_size),int(input_size),2)
            self.filter=np.array(self.filter_img*50)
            self.input_iamge=qimage2ndarray.array2qimage(self.filter)
            self.output_iamge=QPixmap(self.input_iamge)
            self.ui.label_filters_output.setPixmap(self.output_iamge)
            self.ui.label_filters_output.show()
        
        elif self.filters=="Mean":   
                input_size=(self.ui.lineEdit_3.text())
                self.filter_img =self.mean(int(input_size))
                self.filter=np.array(self.filter_img*200)
                self.input_iamge=qimage2ndarray.array2qimage(self.filter)
                self.output_iamge=QPixmap(self.input_iamge)
                self.ui.label_filters_output.setPixmap(self.output_iamge)
                self.ui.label_filters_output.show() 
                 
#       
        elif self.filters=="Median": 
            input_size=(self.ui.lineEdit_3.text())
            self.filter_img =self.median_filter(int(input_size))
            self.filter=np.array(self.filter_img*200)
            self.input_iamge=qimage2ndarray.array2qimage(self.filter)
            self.output_iamge=QPixmap(self.input_iamge)
            self.ui.label_filters_output.setPixmap(self.output_iamge)
            self.ui.label_filters_output.show()  
        
        
        elif self.filters=="Prewitt":    
            self.filter_img =self.prewitt()
            self.filter=np.array(self.filter_img*200)
            self.input_iamge=qimage2ndarray.array2qimage(self.filter)
            self.output_iamge=QPixmap(self.input_iamge)
            self.ui.label_filters_output.setPixmap(self.output_iamge)
            self.ui.label_filters_output.show()
            
              
        elif self.filters=="Roberts":    
            self.filter_img =self.robert()
            self.filter=np.array(self.filter_img*200)
            self.input_iamge=qimage2ndarray.array2qimage(self.filter)
            self.output_iamge=QPixmap(self.input_iamge)
            self.ui.label_filters_output.setPixmap(self.output_iamge)
            self.ui.label_filters_output.show()
            
        elif self.filters=="Sobel":    
            self.filter_img =self.sobel()
            self.filter=np.array(self.filter_img*200)
            self.input_iamge=qimage2ndarray.array2qimage(self.filter)
            self.output_iamge=QPixmap(self.input_iamge)
            self.ui.label_filters_output.setPixmap(self.output_iamge)
            self.ui.label_filters_output.show()  
            
        elif self.filters=="Canny": 
            self.canny_filter()
            self.ui.label_filters_output.setPixmap(QPixmap("canny_edges.jpg"))
  
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
            
        