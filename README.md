# UAV-haze-removal
This code is for haze removal in UAV images, the algorithm is based on dark channel prior and guided-filter.

We change the approach to retrieve the global atmospheric light.

The format of input hazy image is .jpg or .png or .jpeg and the intensity of pixel is 8 bit.


Requirements:

python:3.6x

os

time

opencv-python: pip install opencv-python

numpy: pip install numpy
           HAZY
![https://github.com/Liu-Feng/UAV-haze-removal/blob/master/hazy.png]
          DEHAZED
![https://github.com/Liu-Feng/UAV-haze-removal/blob/master/dehazed.png]
