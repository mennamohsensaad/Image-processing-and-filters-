from PyQt5 import QtWidgets,QtGui , QtCore ,Qt
from PyQt5.QtWidgets import   QFileDialog  ,QWidget,QApplication
from PyQt5.QtGui import QPixmap 
from MainWindow import Ui_MainWindow
from PIL import Image
import matplotlib.pyplot as pl
from PIL.ImageQt import ImageQt
import sys
from os import listdir
from os.path import isfile
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from numpy import asarray
from PIL import Image
from  qimage2ndarray import array2qimage
import seaborn as sns


class Histograms(QtWidgets.QMainWindow):
    def __init__(self):
        super(Histograms, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.draw_curve=0
        self.ui.pushButton_histograms_load.clicked.connect(self.LoadImage)
        self.ui.comboBox_9.currentIndexChanged.connect(self.Draw_histogram)
        self.ui.comboBox_7.currentIndexChanged.connect(self.check_Effect_to_image)
        self.ui.comboBox_8.currentIndexChanged.connect(self.Choose_curve)
       
     
    def LoadImage(self):  
        self.fileName, _filter = QFileDialog.getOpenFileName(self, "Title"," " , "Filter -- img file (*.jpg *.PNG);;img file (*.PNG)")
        if self.fileName:
            pixmap = QPixmap(self.fileName)
            self.pixmap = pixmap.scaled(256,256, QtCore.Qt.KeepAspectRatio, QtCore.Qt.FastTransformation) 
            self.input_img =mpimg.imread(self.fileName)
            self.gray =cv2.imread(self.fileName,0)
            #plt.imshow(self.color_img, cmap=plt.get_cmap('gray'), vmin=0, vmax=1)
            self.ui.label_histograms_input.setPixmap(self.pixmap)
            self.ui.label_histograms_input.show
            #to show size of the image 
            pixels = asarray(self.input_img)
            print(pixels.shape)
            self.ui.lineEdit_4.setText(""+str(pixels.shape[0])+" "+str('x')+" "+str(pixels.shape[1])+"")
        
        
    def rgb2gray(self,rgb_image):
        return np.dot(rgb_image[...,:3], [0.299, 0.587, 0.114])  # ... mean  all rgb values 


#___________________________________________________________________________________________________________
    def Draw_histogram(self):
         self.color_of_histogram = str(self.ui.comboBox_9.currentText())
         print(self.color_of_histogram)
         img = Image.open(self.fileName).convert('YCbCr')
         equized_image=Image.open("equlized_image.jpg").convert('YCbCr')
         #Convert our image to numpy array, calculate the histogram
         self.img = np.array(img)
         self.equized_image = np.array(equized_image)    
         if self.color_of_histogram=="Gray ":
                self.ui.input_histogram.clear()
                self.ui.output_histogram.clear()
                img_arr = np.asarray(self.gray)
                flat = img_arr.flatten()
                gray_hist = self.make_histogram(flat)
                gray_equalized_hist=self.make_histogram(self.new_equalized_img)
                plotWindow = self.ui.input_histogram
                plotWindow.plot(gray_hist, pen='w')
                plotWindow2 = self.ui.output_histogram
                plotWindow2.plot(gray_equalized_hist, pen='w')
    
               
         elif self.color_of_histogram=="Red":
                self.ui.input_histogram.clear()
                self.ui.output_histogram.clear()
            
                # Extract 2-D arrays of the RGB channels: red
                red_pixels=self.img[:,:,0]
                red_pixels_equalized=self.equized_image[:,:,0]
                # Flatten the 2-D arrays into 1-D
                red_vals =  red_pixels.flatten()
                red_vals_equalized =red_pixels_equalized.flatten()
                red_hist = self.make_histogram_of_Color_im(red_vals,self.img)
                red_hist_equalized=self.make_histogram_of_Color_im(red_vals_equalized,self.equized_image)
                plotWindow = self.ui.input_histogram
                plotWindow.plot( red_hist, pen='r')
                plotWindow2 = self.ui.output_histogram
                plotWindow2.plot( red_hist_equalized, pen='r')
    
     
         elif self.color_of_histogram=="Green ":
                # Extract 2-D arrays of the RGB channels: green
                green_pixels=self.img[:,:,1]
                green_pixels_equalized=self.equized_image[:,:,1]
                # Flatten the 2-D arrays into 1-D
                green_vals =green_pixels.flatten()
                green_vals_equalized =green_pixels_equalized.flatten()
                green_hist = self.make_histogram_of_Color_im(green_vals,self.img)
                green_hist_equalized=self.make_histogram_of_Color_im(green_vals_equalized,self.equized_image)
                plotWindow = self.ui.input_histogram
                plotWindow.plot( green_hist, pen='g')
                plotWindow2 = self.ui.output_histogram
                plotWindow2.plot(green_hist_equalized, pen='g')
                
         elif self.color_of_histogram=="Blue ":
                self.ui.input_histogram.clear()
                self.ui.output_histogram.clear()
                
                # Extract 2-D arrays of the RGB channels: blue
                blue_pixels=self.img[:,:,1]
                blue_pixels_equalized=self.equized_image[:,:,1]
                # Flatten the 2-D arrays into 1-D
                blue_vals =blue_pixels.flatten()
                blue_vals_equalized =blue_pixels_equalized.flatten()
                blue_hist = self.make_histogram_of_Color_im(blue_vals,self.img)
                blue_hist_equalized=self.make_histogram_of_Color_im(blue_vals_equalized,self.equized_image)
                plotWindow = self.ui.input_histogram
                plotWindow.plot( blue_hist, pen='b')
                plotWindow2 = self.ui.output_histogram
                plotWindow2.plot(blue_hist_equalized, pen='b')
                
         else :
            self.ui.input_histogram.clear()
            self.ui.output_histogram.clear()
            colors = ('r', 'g', 'b')
            channel_ids = (0, 1, 2)
            # create the histogram plot, with three lines, one for each color
            for channel_id, c in zip(channel_ids, colors):
                                print (channel_id)
                                print(c)
                                # Extract 2-D arrays of the RGB channels: blue
                                color_pixels=self.img[:,:,channel_ids]
                                color_pixels_equalized=equized_image[:,:,channel_ids]    
                                # Flatten the 2-D arrays into 1-D
                                color_vals =color_pixels.flatten()
                                color_vals_equalized =color_pixels_equalized.flatten()
                                color_hist = self.make_histogram_of_Color_im(color_vals,self.img)
                                color_hist_equalized=self.make_histogram_of_Color_im(color_vals_equalized,equized_image)
                                plotWindow = self.ui.input_histogram
                                plotWindow.plot( color_hist, pen=c)
                                plotWindow2 = self.ui.output_histogram
                                plotWindow2.plot(color_hist_equalized, pen=c)
                                
#______________________________________Build histogram______________________________________________
    def make_histogram(self,img):
        # Take a flattened greyscale image and create a historgram from it 
        histogram = np.zeros(256, dtype=int)
        for i in range(img.size):
            histogram[img[i]] += 1
        return histogram                                 

    def make_histogram_of_Color_im(self, y_vals,img):
        """ Take an image and create a historgram from it's luma values """
        histogram = np.zeros(256, dtype=int)
        for y_index in range(y_vals.size):
            histogram[y_vals[y_index]] += 1
        return histogram                                

#______________________________________check options ________________________________________                               
    def check_Effect_to_image(self):
        effect= str(self.ui.comboBox_7.currentText())
        print(effect)
        if effect== "Normalize" :
              self.check_color_or_Gray_Normalize()
        elif effect=="Equalize ":
               self.check_color_or_Gray_Equalize()
        elif effect=="Global Thresholding ":
              thre=(self.ui.lineEdit_10.text())
              thre=float(thre)
              print(thre)
              self.global_threshold(thre)
        else :
             ratio = (self.ui.lineEdit_10.text())
             ratio=float(ratio)
             size = (self.ui.lineEdit_9.text())
             size=int(size)
             self.Local_thresholding(size ,ratio)
            
        
        
    def check_color_or_Gray_Normalize(self):
        pixels = asarray(self.input_img)
        pixels = pixels.astype('float32')
        #check if its RGB or Gray
        try:
            if (pixels.shape[2] == 3):
                self.normalize_color_image(self.input_img)
                print("3")
            
        #elif (pixels.shape[2] == 1):
        except IndexError:  #(pixels.shape[2] == None):
                 self.normalize_grey_image(self.input_img)
                 print("1")
    


    def check_color_or_Gray_Equalize(self):
        pixels = asarray(self.input_img)
        pixels = pixels.astype('float32')
        print(pixels.shape)
        #check if its RGB or Gray
        try:
            if (pixels.shape[2] == 3):
                self.equalize_color_image()
                print("3")
            
        #elif (pixels.shape[2] == 1):
        except IndexError:  #(pixels.shape[2] == None):
                 self.Equilize_grey_Image()
                 print("1")
#______________________________choose curves_____________________________________________________                 
    def  Choose_curve(self):
          curve= str(self.ui.comboBox_8.currentText())
          print(curve)
          if curve== "Cumlative curve " :
              #print("cumaltive")
              self.draw_curve=1
              print(self.draw_curve)
              self.check_RGB_or_Gray_Equalize()
        
          else :
             self.distribution_curve()
             #print("distribution")
             
             
    def  distribution_curve(self):
        img_arr = np.asarray(self.input_img)
        #mg_float = img_arr.astype('float32')
        flat = img_arr.flatten()
        distrubution_curve=sns.distplot(flat, hist=True, kde=True, color = 'darkblue', hist_kws={'edgecolor':'black'},kde_kws={'linewidth': 5})         
        plotWindow2 = self.ui.output_histogram
        plotWindow2.plot(distrubution_curve, pen='w') 
        """note that this is plot in consol """ 
        
    def make_cumsum(self,histogram):
        # Create an array that represents the cumulative sum of the histogram 
        cumsum = np.zeros(256, dtype=int)
        cumsum[0] = histogram[0]
        for i in range(1, histogram.size):
            cumsum[i] = cumsum[i-1] + histogram[i]
        return cumsum     
    def make_cumsum_of_Color_im(self,histogram):
        """ Create an array that represents the cumulative sum of the histogram """
        cumsum = np.zeros(256, dtype=int)
        cumsum[0] = histogram[0]
        for i in range(1, histogram.size):
            cumsum[i] = cumsum[i-1] + histogram[i]
        return cumsum    
            
#______________________________________Normalize image_______________________________________________
    
    def normalize_grey_image(self,img):
        #image = Image.open(img).convert('L')
        pixels = asarray(img)
        #img_h = pixels.shape[0]
        #img_w = pixels.shape[1]
        pixels = pixels.astype('float32')
        
        #need only one as its only one channel
        old_min = pixels.min()
        old_max = pixels.max()
        old_range = old_max - old_min
        
        for rows in range (pixels.shape[0]):
            for col in range (pixels.shape[1]):
                pixels[rows, col]  = (pixels[rows, col] - old_min) / old_range
 
        pixels=np.array(pixels)
        #plt.imshow(pixels)
        img=array2qimage(pixels*255)
        #img.show()
        #img.save('norm_grey_img.png')
        pixmap = QPixmap(img)
        self.pixmap = pixmap.scaled(256, 256, QtCore.Qt.KeepAspectRatio, QtCore.Qt.FastTransformation)
        self.ui.label_histograms_output.setPixmap(self.pixmap)
        self.ui.label_histograms_output.show
    
    
#____________normalize color image________________
        
    def normalize_color_image(self,img):
        
        #read image as array to take size of image
        pixels = asarray(img)
        pixels = pixels.astype('float32')
        #print (pixels.shape)
        #print(pixels.shape[1])
        
        #get minimum and maximum intensity values of image(for each channel) and set a range out of them
        old_minR = pixels[..., 0].min()
        old_minG = pixels[..., 1].min()
        old_minB = pixels[..., 2].min()
        
        old_maxR = pixels[..., 0].max()
        old_maxG = pixels[..., 1].max()
        old_maxB = pixels[..., 2].max()
        
        #Or: max_channels = np.amax([np.amax(img[:,:,0]), np.amax(img[:,:,1]), np.amax(img[:,:,2])])
        
        old_rangeR = old_maxR - old_minR
        old_rangeG = old_maxG - old_minG
        old_rangeB = old_maxB - old_minB
        
        #formula for normalization from (0-255): Inew = (Iold-old_min) * (new_range/old_range) + new_min
        #formula for normalization from (0-1): Inew = (Iold-old_min) /old_range
        
        #for each pixel change its intensity using the formula above
        for rows in range (pixels.shape[0]):
            for col in range (pixels.shape[1]):
                pixels[rows, col,0]  = (pixels[rows, col,0] - old_minR) / old_rangeR
                pixels[rows, col,1]  = (pixels[rows, col,1] - old_minG) / old_rangeG
                pixels[rows, col,2]  = (pixels[rows, col,2] - old_minB) / old_rangeB
                #print(pixels[rows, col])
        
        pixels=np.array(pixels)
        #plt.imshow(pixels)
        iamge=array2qimage(pixels*255)
        pixmap = QPixmap(iamge)
        self.pixmap = pixmap.scaled(256, 256, QtCore.Qt.KeepAspectRatio, QtCore.Qt.FastTransformation)
        self.ui.label_histograms_output.setPixmap(self.pixmap)
        self.ui.label_histograms_output.show
        
    
  
       
#___________________________________________Equalize image______________________________________________________
    def Equilize_grey_Image(self):
        img_arr = np.asarray(self.gray)
        #img_float = img_arr.astype('float32')
        self.img_h = img_arr.shape[0]
        self.img_w = img_arr.shape[1]
        flat = img_arr.flatten()
        #hist = np.histogram(flat, bins=256, range=(0, 1))
        hist = self.make_histogram(flat)
#        plotWindow2 = self.ui.output_histogram
#        plotWindow2.plot(hist, pen='w')
        cumilative_curve = self.make_cumsum(hist)
        if (self.draw_curve==1):
                #print("yyyyyyyyyyyyyes")
                self.ui.output_histogram.clear()
                plotWindow2 = self.ui.output_histogram
                plotWindow2.plot(cumilative_curve, pen='w') 
        else:        
            new_intensity = self.make_mapping(cumilative_curve, self.img_h,self.img_w)
            self.new_equalized_img = self.apply_mapping(flat,new_intensity) #new_img is 1D
            #hist_equ= self.make_histogram(self.new_equalized_img)
      
            self.output_image = Image.fromarray(np.uint8(self.new_equalized_img.reshape((self.img_h,self.img_w))))
            #plt.imshow(self.output_image)
            image=ImageQt(self.output_image)
            self.pix=QPixmap.fromImage(image).scaled(256, 256,QtCore.Qt.KeepAspectRatio, QtCore.Qt.FastTransformation)
            self.ui.label_histograms_output.setPixmap(self.pix)
      
      
    def make_mapping(self,cumsum, img_h, img_w):
        # Create a mapping s.t. each old colour value is mapped to a new
         #   one between 0 and 255 
        mapping = np.zeros(256, dtype=int)
    
        grey_levels = 256
        for i in range(grey_levels):
            mapping[i] = max(0, round((grey_levels*cumsum[i])/(img_h*img_w))-1)
        return mapping

    #create the mapped image
    #new_image[i]=mapping[img[i]]
    #The output of this function is an array containing the pixel values of the new, histogram equalized image! 
    #All that needs doing now is restructuring and rendering / saving it
    def apply_mapping(self,img, mapping):
        # Apply the mapping to our image 
        new_image = np.zeros(img.size, dtype=int)
        for i in range(img.size):
            new_image[i] = mapping[img[i]]
        return new_image
    
#____________equalize color image________________________ 
        
    def equalize_color_image(self):
       # Load image, convert it to YCbCr format ten store width and height into constants
        img = Image.open(self.fileName).convert('YCbCr')
        self.IMG_W, self.IMG_H = img.size
        
        # Convert our image to numpy array, calculate the histogram, cumulative sum,
        # mapping and then apply the mapping to create a new image
        img = np.array(img)
        y_vals = img[:,:,0].flatten()
        histogram = self.make_histogram_of_Color_im( y_vals,img)
#        plotWindow2 = self.ui.output_histogram
#        plotWindow2.plot( histogram, pen='w')
        
        cumilative_curve = self.make_cumsum_of_Color_im(histogram)
        if (self.draw_curve==1):
                #print("yyyyyyyyyyyyyes")
                self.ui.output_histogram.clear()
                plotWindow2 = self.ui.output_histogram
                plotWindow2.plot( cumilative_curve, pen='w') 
        else:      
            mapping = self.make_mapping_of_Color_im(histogram,  cumilative_curve)
            new_image = self.apply_mapping_of_Color_im(img, mapping)
            #new_histogram=self.make_histogram_of_Color_im(new_image)
            #plotWindow2 = self.ui.output_histogram
            #plotWindow2.plot( cumsum, pen='w')
            # Save the image
            self.equalized_color_image = Image.fromarray(np.uint8(new_image), "YCbCr")
            self.equalized_color_image.save("equlized_image.jpg")
            #self.pix=QPixmap.fromImage(image).scaled(256, 256,QtCore.Qt.KeepAspectRatio, QtCore.Qt.FastTransformation)
            self.ui.label_histograms_output.setPixmap(QPixmap("equlized_image.jpg").scaled(256, 256,QtCore.Qt.KeepAspectRatio, QtCore.Qt.FastTransformation))
          
       
    def make_mapping_of_Color_im(self,histogram, cumsum):
        """ Create a mapping s.t. each old luma value is mapped to a new
            one between 0 and 255. Mapping is created using:
             - M(i) = max(0, round((luma_levels*cumsum(i))/(h*w))-1)
            where luma_levels is the number of luma levels in the image """
        mapping = np.zeros(256, dtype=int)
        luma_levels = 256
        for i in range(histogram.size):
            mapping[i] = max(0, round((luma_levels*cumsum[i])/(self.IMG_H*self.IMG_W))-1)
        return mapping
    def apply_mapping_of_Color_im(self,img, mapping):
        """ Apply the mapping to our image """
        new_image = img.copy()
        new_image[:,:,0] = list(map(lambda a : mapping[a], img[:,:,0]))
        return new_image


#_________________________________________global thresholding ____________________________________________
        
    def global_threshold (self,threshold):
        gray_img =cv2.imread(self.fileName,0)  
        img = asarray( gray_img)
        #img = img.astype('float32')
        print(img)
        for row in range (img.shape[0]):
            for col in range (img.shape[1]):
                if img[row, col] < threshold:
                    img[row, col] = 0
                else:
                    img[row, col] = 255
        #new_img = img.astype(np.uint8) #made it save safely
        pixels=np.array(img)
        iamge=array2qimage(pixels)
        #new_img = Image.fromarray(iamge)
        pixmap = QPixmap(iamge)
        self.pixmap = pixmap.scaled(256, 256, QtCore.Qt.KeepAspectRatio, QtCore.Qt.FastTransformation)
        self.ui.label_histograms_output.setPixmap(self.pixmap)
        self.ui.label_histograms_output.show
        threshold=0
        
        
#______________________________________local thresholding_________________________________________        

    def Local_thresholding(self,size,ratio):
         gray_img =cv2.imread(self.fileName,0)  
         image_array = asarray( gray_img)
         #gray_img =self.rgb2gray(self.input_img)
         #pixels = asarray(gray_img)
         #pixels = pixels.astype('float32')
         #image_array = np.array(self.input_img)
         print(image_array)        
         new_array=np.ones(shape=(len(image_array),len(image_array[0])))
         for row  in range( len(image_array)- size + 1 ):
             for col  in range( len(image_array[0]) - size + 1 ):
                 #for row1  in range( len(thelist)- size + 1 ):
                 window=image_array[row:row+size,col:col+size]
                 minm=window.min()
                 maxm=window.max()
                 #print(minm,maxm)
                 threshold =minm+((maxm-minm)*ratio)
                 #print(threshold)
                 if window[0,0] < threshold:
                     new_array[row,col]=0
                     print('ok1')
                        #new_array.append(0)
                     #print ('t')
                    #print('x')      
                 else:
                    new_array[row,col]=1
                    print('ok2')
         print(new_array)           
         pixels=np.array(new_array)
         #gray2qimage
         iamge=array2qimage(pixels*50)
        #new_img = Image.fromarray(iamge)
         pixmap = QPixmap(iamge)
         self.pixmap = pixmap.scaled(256, 256, QtCore.Qt.KeepAspectRatio, QtCore.Qt.FastTransformation)
         self.ui.label_histograms_output.setPixmap(self.pixmap)
         self.ui.label_histograms_output.show
         print(new_array)
         #plt.imshow(new_array, cmap = plt.get_cmap('gray'))    
        
    
        
def main():
    app = QtWidgets.QApplication(sys.argv)
    application = Histograms()
    application.show()
    
  
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()