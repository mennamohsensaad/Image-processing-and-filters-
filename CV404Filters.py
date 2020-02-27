from os import listdir
from os.path import isfile , join
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math


def rgb2gray(rgb_image):
    return np.dot(rgb_image[...,:3], [0.299, 0.587, 0.114])

def multi_view( images ):
    images_count = len( images ) #count of images (list type)
    fig = plt.figure(figsize=(10,10))
    for row in range(images_count  ):
        if row % 2==0:
            ax1 = fig.add_subplot(images_count , 2, row+1)
            ax1.title.set_text('Before')
            ax1.imshow(images[ row ], cmap=plt.get_cmap('gray')) # default is viridis 
            
        else:
            ax1 = fig.add_subplot(images_count , 2, row+1)
            ax1.title.set_text('After')
            ax1.imshow(images[ row ], cmap=plt.get_cmap('gray')) 
           
            
def gaussian_noise( mu, sigma, im_size ):
    randGaussian=np.random.normal( mu, sigma, im_size) #np.random.normal Gaussian noise
    return randGaussian

def im_gaussian_noise(mu, sigma, im):
    g_noise= gaussian_noise(mu,sigma, im.shape)
    img_w_g_noise = im + g_noise
    return img_w_g_noise

        
def Random_Uniform(img,percent):
    img_noisy=np.zeros(img.shape)
    uniform = np.random.random(img.shape) 
    cleanPixels_ind=uniform > percent
    noise = (uniform <= (percent)); 
    img_noisy[cleanPixels_ind]=img[cleanPixels_ind]
    img_noisy[noise] = 0.7
    return img_noisy

def salt_pepper_noise(img,percent):
    img_noisy=np.zeros(img.shape)
    salt_pepper = np.random.random(img.shape) # Uniform distribution
    cleanPixels_ind=salt_pepper > percent
    #NoisePixels_ind=salt_pepper <= percent
    pepper = (salt_pepper <= (0.5* percent)); # pepper < half percent
    
    salt = ((salt_pepper <= percent) & (salt_pepper > 0.5* percent)); 
    img_noisy[cleanPixels_ind]=img[cleanPixels_ind]
    img_noisy[pepper] = 0
    img_noisy[salt] = 1
    return img_noisy

############################## Filters####################################
def corr(img,mask):
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

def gaussian(m,n,sigma):
    gaussian=np.zeros((m,n))
    m=m//2
    n=n//2
    for x in range (-m,m+1):
        for y in range (-n,n+1):
            x1=sigma*math.sqrt(2*np.pi)
            x2=np.exp(-(x**2+y**2)/(2*sigma**2))
            gaussian[x+m,y+n]=(1/x1)*x2  
    return gaussian

def gaussian_filter(m,n,sigma,img):
    g=gaussian(m,n,sigma)
    n=corr(img,g)
    return n
    
def mean(img,k):
    meanFilter=np.ones((k,k))/k*k
    filt=corr(img,meanFilter)
    return filt

def median_filter(img, mask):
    m,n=img.shape
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
                     temp.append(img[i + u - mask_center][j + v - mask_center])

            temp.sort()
            median[i][j] = temp[len(temp) // 2]
            temp = []
    return median  
    

images_files = [ join("./images" , f) for f in listdir("images") if isfile(join("images" , f)) ]
print(images_files)
print(listdir("images"))
images = [ mpimg.imread( f ) for f in images_files ]
gray_images = [ rgb2gray( img ) for img in images ]
result1=[im_gaussian_noise(0, 0.3, img ) for img in gray_images]

result2=[mean(img,5) for img in gray_images] 
result3=[gaussian_filter(5,5,2,img) for img in gray_images] 
result4=[median_filter(img,5) for img in gray_images]  

#print(len(result))
#print(len(result[0][0]))
print("Please,enter 1 to add noise ")
print("       enter 2 to use mean filter ")
print("       enter 3 to use gaussian filter ")
print("       enter 4 to use median filter ")
a=input("Enter : ")
if int(a)== 1:
   # Gaussian
   [multi_view(viewNoise) for viewNoise in (list( zip( gray_images , result1 )))]


   ## Uniform
   uniform_noise = [Random_Uniform (grimages,0.3) for grimages in gray_images]
   [multi_view(viewNoise) for viewNoise in (list( zip( gray_images , uniform_noise )))]

   # Salt & Pepper
   salt_pepper_noise_imgs = [salt_pepper_noise(grimages,0.5) for grimages in gray_images]
   [multi_view(viewNoise) for viewNoise in (list( zip( gray_images , salt_pepper_noise_imgs )))] 

elif int(a)== 2:
    [multi_view(viewNoise) for viewNoise in (list( zip( gray_images , result2 )))]
    
elif int(a)==3:
    [multi_view(viewNoise) for viewNoise in (list( zip( gray_images , result3 )))]
elif int(a)==4:
    [multi_view(viewNoise) for viewNoise in (list( zip( gray_images , result4 )))]
else:
   exit    
    