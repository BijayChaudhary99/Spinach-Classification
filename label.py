import numpy as np 
import sys
import os
from keras.applications.vgg16 import VGG16
import keras
from numpy import load
from matplotlib import pyplot

# from keras.preprocessing.image import load_img
import tensorflow as tf
from keras.utils import load_img,img_to_array


out_path = 'Dataset1/Train'

labels = os.listdir(out_path)
size = [len(os.listdir(os.path.join(out_path, lab))) for lab in labels]

plt.figure(figsize = (10,10))
plt.barh(labels, size, align='center', )
plt.ylabel('Labels')
plt.xlabel('Number of Samples')
plt.title('Size of Lables')
plt.show()
