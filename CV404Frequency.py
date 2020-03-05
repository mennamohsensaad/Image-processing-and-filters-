from PyQt5 import QtWidgets,QtGui , QtCore ,Qt
from PyQt5.QtWidgets import   QFileDialog  ,QWidget,QApplication
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QLabel
from PyQt5.QtGui import QIcon
from MainWindow import Ui_MainWindow
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

class Filters(QtWidgets.QMainWindow):
    def __init__(self):
        super(Filters, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.ui.pushButton_histograms_load_2.clicked.connect(self.button_clicked1)##LOAD IMAGE 1 HYBRID
        self.ui.pushButton_histograms_load_3.clicked.connect(self.button_clicked2)##LOAD IMAGE 2 HYBRID
        self.ui.pushButton_histograms_load_4.clicked.connect(self.button_clicked3)##OUTPUT HYBRID
        #self.ui.pushButton_filters_load.clicked.connect(self.button_clicked)##LOAD IMAGE 2 FILTERS
        #self.ui.comboBox_2.currentIndexChanged.connect(self.Draw_histogram) 

    def button_clicked1(self):  
        fileName, _filter = QFileDialog.getOpenFileName(self, "Title"," " , "Filter -- img file (*.jpg *.PNG);;img file (*.PNG)")
        if fileName:
            pixmap = QPixmap(fileName)
            self.pixmap = pixmap.scaled(256, 256, QtCore.Qt.KeepAspectRatio,QtCore.Qt.FastTransformation) 
            self.color_img1 =mpimg.imread(fileName)
            self.gray_img1 =self.rgb2gray(self.color_img1) 
            
            self.pixels1 = np.asarray(self.color_img1)
            self.pixels1 = self.pixels1.astype('float32')
#            print(self.pixels1.shape)
            
            
          #  marilyn  = ndimage.imread("marilyn.png", flatten=True)
            #plt.imshow(self.color_img, cmap=plt.get_cmap('gray'), vmin=0, vmax=1)
            self.Display_image1()
            self.Label1_Name_Size()
            self.size1()
            
            
    def rgb2gray(self,rgb_image):
        return np.dot(rgb_image[...,:3], [0.299, 0.587, 0.114])  # ... mean  all rgb values     
    
    def button_clicked2(self):  
        fileName, _filter = QFileDialog.getOpenFileName(self, "Title"," " , "Filter -- img file (*.jpg *.PNG);;img file (*.PNG)")
        if fileName:
            pixmap = QPixmap(fileName)
            self.pixmap = pixmap.scaled(256, 256, QtCore.Qt.KeepAspectRatio,QtCore.Qt.FastTransformation) 
            self.color_img2 =mpimg.imread(fileName)
            #plt.imshow(self.color_img, cmap=plt.get_cmap('gray'), vmin=0, vmax=1)
#            einstein = ndimage.imread("einstein.png", flatten=True)
            self.gray_img2 =self.rgb2gray(self.color_img2)
            
            self.pixels2 = np.asarray(self.color_img2)
            self.pixels2 = self.pixels2.astype('float32')
#            print(self.pixels2.shape)
            
            self.Display_image2()
            self.Label2_Name_Size()
            self.size2()
            
    def button_clicked3(self):  

        
           hybrid   = self.hybridImage (self.gray_img2, self.gray_img1, 25, 10)
#           misc.imsave("marilyn-einstein.png", numpy.real(hybrid))
        #       img= Image.open('marilyn-einstein.png')
        #       img.show()
           output_hybird = np.array(numpy.real(hybrid)*200).astype(np.uint8)
           output_hybird = qimage2ndarray.array2qimage(output_hybird)
           output_hybird = QPixmap(output_hybird)
           output_hybird = output_hybird.scaled(256, 256, QtCore.Qt.KeepAspectRatio, QtCore.Qt.FastTransformation)
           
           self.ui.label_histograms_output_2.setPixmap(output_hybird)
           self.ui.label_histograms_output_2.show
            #self.hybridImage()
#           self.Display_image3()        
        
    def Display_image1(self):
        self.ui.label_histograms_input_2.setPixmap(self.pixmap)####for input image 1
        self.ui.label_histograms_input_2.show
    
    def Display_image2(self):
        self.ui.label_histograms_hinput_2.setPixmap(self.pixmap)#####for input image 2
        self.ui.label_histograms_hinput_2.show
        
    def Display_image3(self):
        
        self.ui.label_histograms_output_2.setPixmap(self.pixmap)#####for input image 2
        self.ui.label_histograms_output_2.show     
        
        #label_histograms_output_2 = QLabel(self)
        ##pixmap = QPixmap ('marilyn-einstein.png')
        #label_histograms_output_2.setPixmap(pixmap)
        
    def Label1_Name_Size(self):
        
        #self.ui.label_12 = QLabel(self)
        self.ui.label_12.setText('Name:Marylin')

        
        
    def size1(self):
        self.ui.lineEdit.setText(""+str(self.pixels1.shape[0])+""+str('x')+""+str(self.pixels1.shape[1])+"")

    def size2(self):
        self.ui.lineEdit_2.setText(""+str(self.pixels2.shape[0])+""+str('x')+""+str(self.pixels2.shape[1])+"")
        
      
      
    def Label2_Name_Size(self):
        
        self.ui.label_15.setText('Name:Einstein')
#        self.ui.label_14.setText('Size:256')

#    def scaleSpectrum(self,A):
#        return numpy.real(numpy.log10(numpy.absolute(A) + numpy.ones(A.shape)))


# sample values from a spherical gaussian function from the center of the image
    def makeGaussianFilter(self,numRows, numCols, sigma, highPass=True):
       centerI = int(numRows/2) + 1 if numRows % 2 == 1 else int(numRows/2)
       centerJ = int(numCols/2) + 1 if numCols % 2 == 1 else int(numCols/2)
       def gaussian(i,j):
              coefficient = math.exp(-1.0 * ((i - centerI)**2 + (j - centerJ)**2) / (2 * sigma**2))
              return 1 - coefficient if highPass else coefficient
    
       return numpy.array([[gaussian(i,j) for j in range(numCols)] for i in range(numRows)])

######apply filter by doing coord. multiplication
    def filterFT(self,imageMatrix, filterMatrix):
        ##########apply fourier
       shiftedDFT = fftshift(fft2(imageMatrix))
#       misc.imsave("dft.png", self.scaleSpectrum(shiftedDFT))
       filteredDFT = shiftedDFT * filterMatrix
#       misc.imsave("Lowpassfilter.png", self.scaleSpectrum(filteredDFT))
       return ifft2(ifftshift(filteredDFT))
       
    ####get ride of High freq. comp.(((((((((( Apply fourier to image matrix))))))&&&&& make gaussian for lpf *g(x,y))
    def lowPass(self,imageMatrix, sigma):
       n,m = imageMatrix.shape
       return self.filterFT(imageMatrix, self.makeGaussianFilter(n, m, sigma, highPass=False))
    
    ####get ride of Low freq. comp (((((((((( Apply fourier to image matrix)))))) &&&&&  make gaussian for lpf *1-g(x,y))
    def highPass(self,imageMatrix, sigma):
       n,m = imageMatrix.shape
       return self.filterFT(imageMatrix, self.makeGaussianFilter(n, m, sigma, highPass=True))
    
    ######compining lowpass part of an image with highpass part of another image.
    def hybridImage(self ,highFreqImg, lowFreqImg, sigmaHigh, sigmaLow):
        
       highPassed = self.highPass(highFreqImg, sigmaHigh)
       lowPassed = self.lowPass(lowFreqImg, sigmaLow)    
       return highPassed + lowPassed

#       m = ndimage.imread("marilyn-einstein.png", flatten=True)
#       print(m)
def main():
    app = QtWidgets.QApplication(sys.argv)
    application = Filters()
    application.show()
    
    sys.exit(app.exec_())
      
if __name__ == "__main__":
   main() 
   
#   einstein = ndimage.imread("einstein.png", flatten=True)
#   marilyn  = ndimage.imread("marilyn.png", flatten=True)
