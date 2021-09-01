#!/usr/bin/env python
# coding: utf-8

# In[12]:


import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.feature import hog
import matplotlib.pyplot as plt
import matplotlib.image as mping
import os
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
import pickle


# In[13]:


def gradients_of_pixel_intensity(src):   
    hog_feature1, hog_image1 = hog(src[:, :, 0], orientations=9, pixels_per_cell= (8,8),
        cells_per_block = (2,2),
        block_norm = 'L2-Hys', 
        transform_sqrt=False,
        visualize=True,
        feature_vector=True)

    hog_feature2, hog_image2 = hog(src[:, :, 1], orientations=9, pixels_per_cell= (8,8),
        cells_per_block = (2,2),
        block_norm = 'L2-Hys', 
        transform_sqrt=False,
        visualize=True,
        feature_vector=True)

    hog_feature3, hog_image3 = hog(src[:, :, 2], orientations=9, pixels_per_cell= (8,8),
        cells_per_block = (2,2),
        block_norm = 'L2-Hys', 
        transform_sqrt=False,
        visualize=True,
        feature_vector=True)

    hog_feature = np.concatenate((hog_feature1, hog_feature2, hog_feature3))
    
    return hog_feature


# In[ ]:





# In[ ]:




