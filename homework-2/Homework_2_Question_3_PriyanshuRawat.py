#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
from nuscenes.nuscenes import NuScenes
import os.path as osp

nusc = NuScenes(version='v1.0-mini', dataroot='/users/priya/data/sets/nuscenes/', verbose=True)


# In[2]:


nusc.list_scenes()


# In[3]:


my_scene = nusc.scene[0]
my_scene


# In[4]:


first_sample_token = my_scene['first_sample_token']
print(first_sample_token)


# In[5]:


my_sample = nusc.get('sample', first_sample_token)
my_sample


# In[111]:


nusc.list_sample(my_sample['token'])


# In[112]:


my_sample['data']


# In[113]:


sensor = 'CAM_FRONT'
cam_front_data = nusc.get('sample_data', my_sample['data'][sensor])
cam_front_data


# In[114]:


nusc.render_sample_data(cam_front_data['token'])


# In[115]:


my_annotation_token = my_sample['anns'][18]
my_annotation_metadata =  nusc.get('sample_annotation', my_annotation_token)
my_annotation_metadata


# In[116]:


nusc.render_annotation(my_annotation_token)


# In[117]:


my_instance = nusc.instance[599]
my_instance


# In[118]:


instance_token = my_instance['token']
nusc.render_instance(instance_token)


# In[119]:


print("First annotated sample of this instance:")
nusc.render_annotation(my_instance['first_annotation_token'])


# In[120]:


print("Last annotated sample of this instance")
nusc.render_annotation(my_instance['last_annotation_token'])


# In[121]:


nusc.list_categories()


# In[122]:


nusc.category[9]


# In[123]:


nusc.list_attributes()


# In[124]:


my_instance = nusc.instance[27]
first_token = my_instance['first_annotation_token']
last_token = my_instance['last_annotation_token']
nbr_samples = my_instance['nbr_annotations']
current_token = first_token

i = 0
found_change = False
while current_token != last_token:
    current_ann = nusc.get('sample_annotation', current_token)
    current_attr = nusc.get('attribute', current_ann['attribute_tokens'][0])['name']
    
    if i == 0:
        pass
    elif current_attr != last_attr:
        print("Changed from `{}` to `{}` at timestamp {} out of {} annotated timestamps".format(last_attr, current_attr, i, nbr_samples))
        found_change = True

    next_token = current_ann['next']
    current_token = next_token
    last_attr = current_attr
    i += 1


# In[125]:


nusc.visibility


# In[126]:


anntoken = 'a7d0722bce164f88adf03ada491ea0ba'
visibility_token = nusc.get('sample_annotation', anntoken)['visibility_token']

print("Visibility: {}".format(nusc.get('visibility', visibility_token)))
nusc.render_annotation(anntoken)


# In[127]:


anntoken = '9f450bf6b7454551bbbc9a4c6e74ef2e'
visibility_token = nusc.get('sample_annotation', anntoken)['visibility_token']

print("Visibility: {}".format(nusc.get('visibility', visibility_token)))
nusc.render_annotation(anntoken)


# In[128]:


nusc.sensor


# In[129]:


nusc.sample_data[10]


# In[130]:


nusc.calibrated_sensor[0]


# In[131]:


nusc.ego_pose[0]


# In[132]:


print("Number of `logs` in our loaded database: {}".format(len(nusc.log)))


# In[133]:


nusc.log[0]


# In[134]:


print("There are {} maps masks in the loaded dataset".format(len(nusc.map)))


# In[135]:


nusc.map[0]


# In[ ]:




