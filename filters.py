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
        #self.ui.comboBox.currentIndexChanged.connect(self.show_filters)
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
        
    def corr(self,img,mask):
        row,col=img.shape
        m,n=mask.shape
        new=np.zeros((row+m-1,col+n-1))
        n=n//2
        m=m//2
        filtered=np.zeros(img.shape)
        new[m:new.shape[0]-m,n:new.shape[1]-n]=img
        for i in range (m,new.shape[0]-m):
            for j in range (n,new.shape[1]-n):
                temp=new[i-m:i+m+1,j-m:j+m+1]
                result=temp*mask
                filtered[i-m,j-n]=result.sum()       
        return filtered

    def gaussian(self,m,n,sigma):
        self.gaussian=np.zeros((m,n))
        m=m//2
        n=n//2
        for x in range (-m,m+1):
            for y in range (-n,n+1):
                x1=sigma*math.sqrt(2*np.pi)
                x2=np.exp(-(x**2+y**2)/(2*sigma**2))
                self.gaussian[x+m,y+n]=(1/x1)*x2  
        return self.gaussian

    def gaussian_filter(self,m,n,sigma,img):
        g=self.gaussian(m,n,sigma)
        n=self.corr(img,g)
        return n
    
    def mean(self,img,k):
        meanFilter=np.ones((k,k))/k*k
        filt=self.corr(img,meanFilter)
        return filt
##    
    def show_filters(self): 
        self.filters = str(self.ui.comboBox_2.currentText())
        
        if self.filters=="Gaussian":
            #self.input_iamge=array2qimage.qimage2ndarray(self.input_iamge)
            self.filter_img =self.gaussian_filter(5,5,2,self.self.gray_img)
            self.input_iamge=qimage2ndarray.array2qimage(self.filter_img)
            self.output_iamge=QPixmap(self.input_iamge)
            self.ui.label_filters_output.setPixmap(self.output_iamge)
            self.ui.label_filters_output.show()
        
        elif self.filters=="Mean":    
            self.filter_img =self.mean(self.input_iamge,5)
            self.input_iamge=qimage2ndarray.array2qimage(self.filter_img)
            self.output_iamge=QPixmap(self.input_iamge)
            self.ui.label_filters_output.setPixmap(self.output_iamge)
            self.ui.label_filters_output.show()      
#       
        else:
            print("2")
    #def add_noise(self):
        





def main():
    app = QtWidgets.QApplication(sys.argv)
    application = Filters()
    application.show()
    
  
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()        
            
        