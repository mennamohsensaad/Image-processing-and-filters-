
from MainWindow_Filters import Ui_MainWindow
from scipy import ndimage
from scipy.ndimage.filters import convolve
from scipy import misc
import numpy as np
from PyQt5 import QtWidgets,QtGui , QtCore ,Qt
from PyQt5.QtWidgets import   QFileDialog  ,QWidget,QApplication
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QLabel
from PyQt5.QtGui import QIcon
from PIL import Image
import matplotlib.pyplot as pl
from PIL.ImageQt import ImageQt
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import QLineEdit
from PyQt5.QtWidgets import QLabel
import sys
from os import listdir
from os.path import isfile
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy
from numpy.fft import fft2, ifft2, fftshift, ifftshift
from scipy import misc
from scipy import ndimage
import math
from scipy.misc import imsave
from matplotlib.pyplot import imread
from imageio import imread
import qimage2ndarray


#class cannyEdgeDetector(QtWidgets.QMainWindow):
#    
class Filters (QtWidgets.QMainWindow):
    def __init__(self, sigma=1, kernel_size=5, weak_pixel=75, strong_pixel=255, lowthreshold=0.05, highthreshold=0.15):

        
        
        super(Filters, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.ui.pushButton_filters_load.clicked.connect(self.button_clicked1)##LOAD IMAGE 1 HYBRID
        
        
        self.imgs_final = []
        self.img_smoothed = None
        self.gradientMat = None
        self.thetaMat = None
        self.nonMaxImg = None
        self.thresholdImg = None
        self.weak_pixel = weak_pixel
        self.strong_pixel = strong_pixel
        self.sigma = sigma
        self.kernel_size = kernel_size
        self.lowThreshold = lowthreshold
        self.highThreshold = highthreshold
        return 
    

    def button_clicked1(self):  
        fileName, _filter = QFileDialog.getOpenFileName(self, "Title"," " , "Filter -- img file (*.jpg *.PNG);;img file (*.PNG)")
        if fileName:
            pixmap = QPixmap(fileName)
            self.pixmap = pixmap.scaled(256, 256, QtCore.Qt.KeepAspectRatio,QtCore.Qt.FastTransformation) 
            self.color_img =mpimg.imread(fileName)
            self.gray_img =self.rgb2gray(self.color_img) 
            
            self.Input_Image = np.asarray(self.color_img)
            self.Image = self.Input_Image.astype('float32')
            print(self.Input_Image.shape)
            
            
            self.Label_Name()
            self.size()
            self.Display_image1() 
            
            
            self.output = np.asarray(numpy.real(self.imgs_final)).astype(np.uint8)
            self.output= qimage2ndarray.array2qimage(self.output)
            self.output= QPixmap(self.output)
            self.output = self.output.scaled(256, 256, QtCore.Qt.KeepAspectRatio, QtCore.Qt.FastTransformation)
            
            self.ui.label_filters_output.setPixmap(self.output)
            self.ui.label_filters_output.show
            
            
    def Display_image1(self): 
        self.ui.label_filters_input.setPixmap(self.pixmap)####for input image 1
        self.ui.label_filters_input.show
        
        
        
    def Label_Name(self):
        self.ui.label.setText('Name:FaceImage')  
        
        
    def size(self):
        self.ui.lineEdit_3.setText(""+str(self.Input_Image.shape[0])+""+str('x')+""+str(self.Input_Image.shape[1])+"")
        
#            print(self.pixels1.shape)
    def rgb2gray(self,rgb_image):
        return np.dot(rgb_image[...,:3], [0.299, 0.587, 0.114])  # ... mean  all rgb values     
    
    
    
    
    
    
    
    
    def gaussian_kernel(self, size, sigma=1):
        size = int(size) // 2
        x, y = np.mgrid[-size:size+1, -size:size+1]
        normal = 1 / (2.0 * np.pi * sigma**2)
        g =  np.exp(-((x**2 + y**2) / (2.0*sigma**2))) * normal
        return g
    
    def sobel_filters(self, img):
        Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
        Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)

        Ix = ndimage.filters.convolve(img, Kx)
        Iy = ndimage.filters.convolve(img, Ky)

        G = np.hypot(Ix, Iy)
        G = G / G.max() * 255
        theta = np.arctan2(Iy, Ix)
        return (G, theta)
    

    def non_max_suppression(self, img, D):
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

    def threshold(self, img):

        highThreshold = img.max() * self.highThreshold;
        lowThreshold = highThreshold * self.lowThreshold;

        M, N = img.shape
        res = np.zeros((M,N), dtype=np.int32)

        weak = np.int32(self.weak_pixel)
        strong = np.int32(self.strong_pixel)

        strong_i, strong_j = np.where(img >= highThreshold)
        zeros_i, zeros_j = np.where(img < lowThreshold)

        weak_i, weak_j = np.where((img <= highThreshold) & (img >= lowThreshold))

        res[strong_i, strong_j] = strong
        res[weak_i, weak_j] = weak

        return (res)

    def hysteresis(self, img):

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
    
    def detect(self):
        self.imgs_final = []
        for i, img in enumerate(self.imgs):    
            self.img_smoothed = convolve(img, self.gaussian_kernel(self.kernel_size, self.sigma))
            self.gradientMat, self.thetaMat = self.sobel_filters(self.img_smoothed)
            self.nonMaxImg = self.non_max_suppression(self.gradientMat, self.thetaMat)
            self.thresholdImg = self.threshold(self.nonMaxImg)
            img_final = self.hysteresis(self.thresholdImg)
            self.imgs_final.append(img_final)

        return self.imgs_final
#        img=self.imgs_final
#        img.show()
        
    
def main():
    app = QtWidgets.QApplication(sys.argv)
    application = Filters()
    application.show()
    
    sys.exit(app.exec_())
      
if __name__ == "__main__":
   main() 