"""
visualize results for test image
"""

import numpy as np

from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from torch.autograd import Variable

import transforms as transforms
from skimage import io
from skimage.transform import resize
from models import *

cut_size = 44

transform_test = transforms.Compose([
    transforms.TenCrop(cut_size),
    transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
])

read_img = io.imread('images/1.jpg', as_gray=True)  # raw frames to read, convert to grayscale
resized_img = resize(read_img, (48,48), mode='symmetric').astype(np.uint8)

img = resized_img[:, :, np.newaxis]
img = np.concatenate((img, img, img), axis=2)
img = Image.fromarray(img)
inputs = transform_test(img)

class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

net = VGG('VGG19')
checkpoint = torch.load('../FER_pytorch/PrivateTest_model.t7',     # path to model
    map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
net.load_state_dict(checkpoint['net'])
net.eval()

ncrops, c, h, w = np.shape(inputs)
inputs = inputs.view(-1, c, h, w)
outputs = net(inputs)
outputs_avg = outputs.view(ncrops, -1).mean(0)  # avg over crops

score = F.softmax(outputs_avg)
_, predicted = torch.max(outputs_avg.data, 0)
print(score)
print(predicted)
<<<<<<< HEAD
print("The Expression is %s" %str(class_names[int(predicted.cpu().numpy())]))
=======
print("The Expression is %s" %str(class_names[int(predicted.cpu().numpy())]))
>>>>>>> 5ae64ce56ad091b1f6c19b7061014814caf4d644
