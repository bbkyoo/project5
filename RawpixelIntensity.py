#!/usr/bin/env python
# coding: utf-8

# In[3]:


import cv2
import numpy as np


# In[4]:


def raw_pixel_intensity(src, idx):
    src_r = cv2.resize(src, (idx,idx))
    src_r = src_r.flatten()
    
    return src_r 


# In[ ]:




