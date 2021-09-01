#!/usr/bin/env python
# coding: utf-8

# In[3]:


import cv2
import numpy as np


# In[4]:


def histogram_of_pixel_intensity(src, size): 
    hist1 = cv2.calcHist(images=[src], channels=[0], mask=None, histSize=[size], ranges=[0, 256])
    hist2 = cv2.calcHist(images=[src], channels=[1], mask=None, histSize=[size], ranges=[0, 256])
    hist3 = cv2.calcHist(images=[src], channels=[2], mask=None, histSize=[size], ranges=[0, 256])

    hist1 = hist1.flatten()
    hist2 = hist2.flatten()
    hist3 = hist3.flatten()

    hist = np.concatenate((hist1,hist2,hist3))
    return hist


# In[ ]:




