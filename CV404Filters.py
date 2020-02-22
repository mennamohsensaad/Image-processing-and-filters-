from os import listdir
from os.path import isfile , join
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy import ndimage, signal
from skimage import filters

def rgb2gray(rgb_image):
    return np.dot(rgb_image[...,:3], [0.299, 0.587, 0.114])

def multi_view( images ):
    images_count = len( images ) #count of images (list type)
    fig = plt.figure(figsize=(10,10))
    for row in range(images_count  ):
        if row % 2==0:
            ax1 = fig.add_subplot(images_count , 2, row+1)
            ax1.title.set_text('Before noise')
        #ax1.imshow( images[ row ] )
            ax1.imshow(images[ row ], cmap=plt.get_cmap('gray')) # default is viridis 
            
        else:
            ax1 = fig.add_subplot(images_count , 2, row+1)
            ax1.title.set_text('After noise')
        #ax1.imshow( images[ row ] )
            ax1.imshow(images[ row ], cmap=plt.get_cmap('gray')) # default is viridis 
            
def gaussian_noise( mu, sigma, im_size ):
    randGaussian=np.random.normal( mu, sigma, im_size) #np.random.normal Gaussian noise
    #print(randUniform) 
    #print(np.mean(randUniform))
    #print(np.std(randUniform))   
    #print('\n')
    
    return randGaussian

def im_gaussian_noise(mu, sigma, im):
    g_noise= gaussian_noise(mu,sigma, im.shape)
    img_w_g_noise = im + g_noise
    return img_w_g_noise

def Uniform(img):
    uniform=np.random.random(img.shape)
    return uniform            
        

def salt_pepper_noise(img,percent):
    img_noisy=np.zeros(img.shape)
    salt_pepper = np.random.random(img.shape) # Uniform distribution
    
    #print(img.size)
    #print(salt_pepper.size)
    #print((salt_pepper > percent)*1)
    #print(np.sum((salt_pepper > percent)*1)/img.size) #percent clean
    cleanPixels_ind=salt_pepper > percent
    NoisePixels_ind=salt_pepper <= percent
    pepper = (salt_pepper <= (0.5* percent)); # pepper < half percent
    
    salt = ((salt_pepper <= percent) & (salt_pepper > 0.5* percent)); 
    
    #print(np.sum((salt)*1)/img.size)
    #print(cleanPixels_ind.shape)
    #print(img_noisy.shape)
    img_noisy[cleanPixels_ind]=img[cleanPixels_ind]
    img_noisy[pepper] = 0
    img_noisy[salt] = 1
    return img_noisy

images_files = [ join("./images" , f) for f in listdir("images") if isfile(join("images" , f)) ]
print(images_files)
print(listdir("images"))
images = [ mpimg.imread( f ) for f in images_files ]
gray_images = [ rgb2gray( img ) for img in images ]
result=[im_gaussian_noise(0, 0.3, img ) for img in gray_images] 

print(len(result))
print(len(result[0][0]))

# Gaussian
[multi_view(viewNoise) for viewNoise in (list( zip( gray_images , result )))]

## Uniform
uniform_noise = [Uniform(grimages) for grimages in gray_images]
[multi_view(viewNoise) for viewNoise in (list( zip( gray_images , uniform_noise )))]

# Salt & Pepper
salt_pepper_noise_imgs = [salt_pepper_noise(grimages,0.5) for grimages in gray_images]
[multi_view(viewNoise) for viewNoise in (list( zip( gray_images , salt_pepper_noise_imgs )))]