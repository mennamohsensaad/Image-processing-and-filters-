#import numpy as np
#import skimage
#import matplotlib.pyplot as plt 
#import matplotlib.image as mpimg
#import os
#import scipy.misc as sm
#
#def rgb2gray(rgb):
#
#    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
#    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
#
#    return gray
#
#def load_data(dir_name = 'faces_imgs'):    
#    '''
#    Load images from the "faces_imgs" directory
#    Images are in JPG and we convert it to gray scale images
#    '''
#    imgs = []
#    for filename in os.listdir(dir_name):
#        if os.path.isfile(dir_name + '/' + filename):
#            img = mpimg.imread(dir_name + '/' + filename)
#            img = rgb2gray(img)
#            imgs.append(img)
#    return imgs
#
#
#def visualize(imgs, format=None, gray=False):
#    plt.figure(figsize=(20, 40))
#    for i, img in enumerate(imgs):
#        if img.shape[0] == 3:
#            img = img.transpose(1,2,0)
#        plt_idx = i+1
#        plt.subplot(2, 2, plt_idx)
#        plt.imshow(img, format)
#    plt.show()
    

import cv2
import argparse
from scipy import ndimage
from scipy.ndimage.filters import convolve
from matplotlib.image import imread
import matplotlib.pyplot as plt
import numpy as np
from scipy import misc
 
#from Computer_Vision.Canny_Edge_Detection.sobel import sobel_edge_detection
#from Computer_Vision.Canny_Edge_Detection.gaussian_smoothing import gaussian_blur
 
import matplotlib.pyplot as plt
 
def gaussian_kernel(size, sigma):
        size = int(size) // 2
        x, y = np.mgrid[-size:size+1, -size:size+1]
        normal = 1 / (2.0 * np.pi * sigma**2)
        g =  np.exp(-((x**2 + y**2) / (2.0*sigma**2))) * normal
        return g
    
def sobel_filters(img):
        Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
        Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)

        Ix = ndimage.filters.convolve(img, Kx)
        Iy = ndimage.filters.convolve(img, Ky)

        G = np.hypot(Ix, Iy)
        G = G / G.max() * 255
        theta = np.arctan2(Iy, Ix)
        return (G, theta)
    
def non_max_suppression(gradient_magnitude, gradient_direction, verbose):
    image_row, image_col = gradient_magnitude.shape
 
    output = np.zeros(gradient_magnitude.shape)
 
    PI = 180
 
    for row in range(1, image_row - 1):
        for col in range(1, image_col - 1):
            direction = gradient_direction[row, col]
 
            if (0 <= direction < PI / 8) or (15 * PI / 8 <= direction <= 2 * PI):
                before_pixel = gradient_magnitude[row, col - 1]
                after_pixel = gradient_magnitude[row, col + 1]
 
            elif (PI / 8 <= direction < 3 * PI / 8) or (9 * PI / 8 <= direction < 11 * PI / 8):
                before_pixel = gradient_magnitude[row + 1, col - 1]
                after_pixel = gradient_magnitude[row - 1, col + 1]
 
            elif (3 * PI / 8 <= direction < 5 * PI / 8) or (11 * PI / 8 <= direction < 13 * PI / 8):
                before_pixel = gradient_magnitude[row - 1, col]
                after_pixel = gradient_magnitude[row + 1, col]
 
            else:
                before_pixel = gradient_magnitude[row - 1, col - 1]
                after_pixel = gradient_magnitude[row + 1, col + 1]
 
            if gradient_magnitude[row, col] >= before_pixel and gradient_magnitude[row, col] >= after_pixel:
                output[row, col] = gradient_magnitude[row, col]
 
    if verbose:
        plt.imshow(output, cmap='gray')
        plt.title("Non Max Suppression")
        plt.show()
 
    return output
 
 
def threshold(image, low, high, weak, verbose=False):
    output = np.zeros(image.shape)
 
    strong = 255
 
    strong_row, strong_col = np.where(image >= high)
    weak_row, weak_col = np.where((image <= high) & (image >= low))
 
    output[strong_row, strong_col] = strong
    output[weak_row, weak_col] = weak
 
    if verbose:
        plt.imshow(output, cmap='gray')
        plt.title("threshold")
        plt.show()
 
    return output
 
 
def hysteresis(image, weak):
    image_row, image_col = image.shape
 
    top_to_bottom = image.copy()
 
    for row in range(1, image_row):
        for col in range(1, image_col):
            if top_to_bottom[row, col] == weak:
                if top_to_bottom[row, col + 1] == 255 or top_to_bottom[row, col - 1] == 255 or top_to_bottom[row - 1, col] == 255 or top_to_bottom[
                    row + 1, col] == 255 or top_to_bottom[
                    row - 1, col - 1] == 255 or top_to_bottom[row + 1, col - 1] == 255 or top_to_bottom[row - 1, col + 1] == 255 or top_to_bottom[
                    row + 1, col + 1] == 255:
                    top_to_bottom[row, col] = 255
                else:
                    top_to_bottom[row, col] = 0
 
    bottom_to_top = image.copy()
 
    for row in range(image_row - 1, 0, -1):
        for col in range(image_col - 1, 0, -1):
            if bottom_to_top[row, col] == weak:
                if bottom_to_top[row, col + 1] == 255 or bottom_to_top[row, col - 1] == 255 or bottom_to_top[row - 1, col] == 255 or bottom_to_top[
                    row + 1, col] == 255 or bottom_to_top[
                    row - 1, col - 1] == 255 or bottom_to_top[row + 1, col - 1] == 255 or bottom_to_top[row - 1, col + 1] == 255 or bottom_to_top[
                    row + 1, col + 1] == 255:
                    bottom_to_top[row, col] = 255
                else:
                    bottom_to_top[row, col] = 0
 
    right_to_left = image.copy()
 
    for row in range(1, image_row):
        for col in range(image_col - 1, 0, -1):
            if right_to_left[row, col] == weak:
                if right_to_left[row, col + 1] == 255 or right_to_left[row, col - 1] == 255 or right_to_left[row - 1, col] == 255 or right_to_left[
                    row + 1, col] == 255 or right_to_left[
                    row - 1, col - 1] == 255 or right_to_left[row + 1, col - 1] == 255 or right_to_left[row - 1, col + 1] == 255 or right_to_left[
                    row + 1, col + 1] == 255:
                    right_to_left[row, col] = 255
                else:
                    right_to_left[row, col] = 0
 
    left_to_right = image.copy()
 
    for row in range(image_row - 1, 0, -1):
        for col in range(1, image_col):
            if left_to_right[row, col] == weak:
                if left_to_right[row, col + 1] == 255 or left_to_right[row, col - 1] == 255 or left_to_right[row - 1, col] == 255 or left_to_right[
                    row + 1, col] == 255 or left_to_right[
                    row - 1, col - 1] == 255 or left_to_right[row + 1, col - 1] == 255 or left_to_right[row - 1, col + 1] == 255 or left_to_right[
                    row + 1, col + 1] == 255:
                    left_to_right[row, col] = 255
                else:
                    left_to_right[row, col] = 0
 
    final_image = top_to_bottom + bottom_to_top + right_to_left + left_to_right
 
    final_image[final_image > 255] = 255
 
    return final_image
 
 
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True, help="Path to the image")
    ap.add_argument("-v", "--verbose", type=bool, default=False, help="Path to the image")
    args = vars(ap.parse_args())
 
    image = cv2.imread(args["image"])
 
    blurred_image = gaussian_kernel(5,1)
 
    edge_filter = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
 
    gradient_magnitude, gradient_direction = sobel_filters(blurred_image, edge_filter, convert_to_degree=True, verbose=args["verbose"])
 
    new_image = non_max_suppression(gradient_magnitude, gradient_direction, verbose=args["verbose"])
 
    weak = 50
 
    new_image = threshold(new_image, 5, 20, weak=weak, verbose=args["verbose"])
 
    new_image = hysteresis(new_image, weak)
 
    plt.imshow(new_image, cmap='gray')
    plt.title("Canny Edge Detector")
    plt.show()




