#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os


# In[2]:


def load_dataset(vehicles_path = "./vehicles/vehicles", non_vehicles_path = "./non-vehicles/non-vehicles"):
    #vehicles_path = "./vehicles/vehicles"
    #non_vehicles_path = "./non-vehicles/non-vehicles"
    vehicles_folder_name = os.listdir(vehicles_path)
    non_vehicles_folder_name = os.listdir(non_vehicles_path)

    vehicles_dataset = []
    non_vehicles_dataset = []

    for i in vehicles_folder_name:
        tmp = os.listdir('./vehicles/vehicles/{}'.format(i))
        tmp2 = []
        for j in range(len(tmp)):
            tmp2.append(vehicles_path + '/' + i + '/' + tmp[j])
        vehicles_dataset = vehicles_dataset + tmp2

    for i in non_vehicles_folder_name:
        tmp = os.listdir('./non-vehicles/non-vehicles/{}'.format(i))
        tmp2 = []
        for j in range(len(tmp)):
            tmp2.append(non_vehicles_path + '/' + 'i + '/' + tmp[j])
        non_vehicles_dataset = non_vehicles_dataset + tmp2

    name_dataset = vehicles_dataset + non_vehicles_dataset

    num_vehicles_dataset = [1 for i in range(len(vehicles_dataset))]
    num_non_vehicles_dataset = [0 for i in range(len(non_vehicles_dataset))]
    sum_dataset = num_vehicles_dataset + num_non_vehicles_dataset
    
    return vehicles_dataset, non_vehicles_dataset, name_dataset, num_vehicles_dataset, num_non_vehicles_dataset, sum_dataset


# In[ ]:




