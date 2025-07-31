Face Mask Detection Dataset
This repository contains a Jupyter Notebook for face mask detection.

Setup
To get started, you'll need to install the kaggle library:

Python

!pip install kaggle
Dataset Extraction
The dataset is compressed in a zip file. Use the following code to extract it:

Python

#extracting the compressed dataset
from zipfile import ZipFile
dataset = '/content/face_mask_dataset.zip'

with ZipFile(dataset,'r') as zip:
  zip.extractall()
  print('The dataset is extracted')
You can verify the extracted files using ls:

Bash

!ls
Expected output:

data  face_mask_dataset.zip  sample_data
Importing Dependencies
The project relies on the following Python libraries:

Python

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
import cv2
from google.colab.patches import cv2_imshow
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
Dataset Overview
The dataset consists of images categorized into with_mask and without_mask directories.

To see some of the file names and the count of images in each category:

Python

with_mask_files = os.listdir('/content/data/with_mask')
without_mask_files = os.listdir('/content/data/without_mask')
print(with_mask_files[:10])
print(without_mask_files[:10])
print('Number of with mask images:', len(with_mask_files))
print('Number of without mask images:', len(without_mask_files))
Example output:

['with_mask_469.jpg', 'with_mask_1442.jpg', 'with_mask_3179.jpg', 'with_mask_2649.jpg', 'with_mask_38.jpg', 'with_mask_1473.jpg', 'with_mask_2296.jpg', 'with_mask_2457.jpg', 'with_mask_1999.jpg', 'with_mask_732.jpg']
['without_mask_3272.jpg', 'without_mask_3207.jpg', 'without_mask_3582.jpg', 'without_mask_271.jpg', 'without_mask_3072.jpg', 'without_mask_523.jpg', 'without_mask_2435.jpg', 'without_mask_340.jpg', 'without_mask_1918.jpg', 'without_mask_1292.jpg']
Number of with mask images: 3725
Number of without mask images: 3828
Creating Labels
Labels are created for the two classes of images:

with mask: 1

without mask: 0

Python

with_mask_labels = [1]*3725
without_mask_labels = [0]*3828
print(with_mask_labels[:10])
print(without_mask_labels[:10])
print(len(with_mask_labels))
print(len(without_mask_labels))
labels = with_mask_labels + without_mask_labels
print(len(labels))
Displaying Images
You can display an example image with a mask using the following code:

Python

img = mpimg.imread('/content/data/with_mask/with_mask_2682.jpg')
imgplot = plt.imshow(img)
plt.show()
