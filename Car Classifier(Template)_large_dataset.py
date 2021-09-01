#!/usr/bin/env python
# coding: utf-8

# ## Imports

# In[1]:


import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.feature import hog
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
import pickle


# In[2]:


import LoadDataset
import RawpixelIntensity
import HistogramOfPixelIntensity
import GradientsOfPixelIntensity


# ## Load Dataset

# In[3]:


vehicles_path = "./vehicles/vehicles"
non_vehicles_path = "./non-vehicles/non-vehicles"
vehicles_dataset, non_vehicles_dataset, name_dataset, num_vehicles_dataset, num_non_vehicles_dataset, sum_dataset = LoadDataset.load_dataset(vehicles_path, non_vehicles_path)


# ## Basic Summary of Dataset

# In[4]:


tmp1 = pd.DataFrame(name_dataset)
tmp2 = pd.DataFrame(sum_dataset)


# In[5]:


tmp1.shape
tmp2.shape


# In[6]:


tmp1.describe


# In[7]:


tmp2.describe


# ## Visualize Some of the Data

# In[31]:


fig, axs = plt.subplots(8,8, figsize=(16, 16))
fig.subplots_adjust(hspace = .2, wspace=.001)
axs = axs.ravel()

# Step through the list and search for chessboard corners
for i in np.arange(32):
    img = mpimg.imread(vehicles_dataset[np.random.randint(0,len(vehicles_dataset))])
    #img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    axs[i].axis('off')
    axs[i].set_title('car', fontsize=10)
    axs[i].imshow(img)
for i in np.arange(32,64):
    img = mpimg.imread(non_vehicles_dataset[np.random.randint(0,len(non_vehicles_dataset))])
    #img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    axs[i].axis('off')
    axs[i].set_title('noncar', fontsize=10)
    axs[i].imshow(img)


# ## Feature Extraction Method

# In[9]:


# Raw pixel intensity : (Color and Shape)
# Histogram of pixel intensity : (Color only)
# Gradients of pixel intensity : (Shape only)


# ## Feature Extraction

# In[16]:


orient = 9
pix_per_cell = 8
cell_per_block = 2


# In[18]:


features = []
spatial = 32
hist_bins = 32

for i in vehicles_dataset:
    src = mpimg.imread(i)
    src = cv2.cvtColor(src, cv2.COLOR_RGB2YCrCb)

    
    raw = RawpixelIntensity.raw_pixel_intensity(src, spatial)
    hist = HistogramOfPixelIntensity.histogram_of_pixel_intensity(src, hist_bins)
    grad = GradientsOfPixelIntensity.gradients_of_pixel_intensity(src)        
    features.append(np.concatenate((raw, hist, grad)))

for i in non_vehicles_dataset:
    src = mpimg.imread(i)
    src = cv2.cvtColor(src, cv2.COLOR_RGB2YCrCb)

    raw = RawpixelIntensity.raw_pixel_intensity(src, spatial)
    hist = HistogramOfPixelIntensity.histogram_of_pixel_intensity(src, hist_bins)
    grad = GradientsOfPixelIntensity.gradients_of_pixel_intensity(src)      
    features.append(np.concatenate((raw, hist, grad)))


# In[19]:


features = np.array(features)


# ## Data Preparation

# In[20]:


scaler = StandardScaler()
X_scaler = scaler.fit_transform(features)


# In[21]:


X_train, X_test, y_train, y_test = train_test_split(X_scaler, sum_dataset)


# In[22]:


X_test.shape


# ## Classifier

# In[23]:


clf = LinearSVC()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
accuracy_score(y_test, y_pred)


# - 500개 정도에서의 정확도(채널별)

# In[24]:


clf = LinearSVC()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
accuracy_score(y_test, y_pred)


# - 채널별 정확도

# In[25]:


clf = LinearSVC()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
accuracy_score(y_test, y_pred)


# - grayScale에서 grid

# In[26]:


pamran_grid = [
    { 'max_iter':[10000]}
]

grid_search = GridSearchCV(clf, pamran_grid, cv=5, scoring = "accuracy", n_jobs=1)
grid_search.fit(X_train, y_train)


# In[27]:


grid_search.best_params_


# In[28]:


grid_search.best_score_


# ## Data Saving to Pickle

# In[29]:


dist_pickle = {}
dist_pickle["svc"] = clf
dist_pickle["scaler"] = X_scaler
dist_pickle["orient"] = orient
dist_pickle["pix_per_cell"] = pix_per_cell
dist_pickle["cell_per_block"] = cell_per_block
dist_pickle["spatial"] = spatial # => resize 조정할때
dist_pickle["hist_bins"] = hist_bins
pickle.dump(dist_pickle, open("svc_pickle.p", 'wb'))


# In[ ]:




