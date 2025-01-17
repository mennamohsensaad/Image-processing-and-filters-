# Image processing and filters


## Objectives

* Filtration of noisy images using low pass filters such as: average, Gaussian, median.
* Edge detection using variety of masks such as: Sobel, Prewitt, and canny edge detectors.
* Histograms and equalization.
* Frequency domain filters.
* Hybrid images.



### A) Computer Vision Functions

need to implement Python functions which will support the following tasks:

1. Add additive noise to the image.
    * For example: Uniform, Gaussian and salt & pepper noise.
2. Filter the noisy image using the following low pass filters.
    * Average, Gaussian and median filters.
3. Detect edges in the image using the following masks
    * Sobel, Roberts , Prewitt and canny edge detectors.
4. Draw histogram and distribution curve.
5. Equalize the image.
6. Normalize the image.
7. Local and global thresholding.
8. Transformation from color image to gray scale image and plot of R, G, and B histograms with its distribution function (cumulative curve that you use it for mapping and histogram equalization).
9. Frequency domain filters (high pass and low pass).
10. Hybrid images.


Organize implementation among the following files:

1. `CV404Filters.py`: this will include your implementation for filtration functions (requirements 1-3).
2. `CV404Histograms.py`: this will include your implementation for histogram related tasks (requirements 4-8).
3. `CV404Frequency.py`: this will include your implementation for frequency domain related tasks (requirements 9-10).

### B) GUI Integration

Integrate  functions in part (A) to the following Qt MainWindow design:

| Tab 1 |
|---|
| <img src=".screen/tab1.png" style="width:500px"> 

| Tab 2 |
|---|
| <img src=".screen/tab2.png" style="width:500px;"> |

| Tab 3 |
|---|
| <img src=".screen/tab3.png" style="width:500px;"> |



## link for demo :- https://drive.google.com/file/d/1C1aRlzgId0_spkh9iYVtojZs0Dd1hDId/view?usp=drivesdk
