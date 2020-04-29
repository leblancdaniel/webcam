"""
Emotion Detector for object tracking pipeline
"""
import os
import numpy as np

from tracking import Image
from tracking.FER import VGG
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import albumentations as albu 

class EmotionDetector:
    def __init__(self):
        self.cut_size = 44

    def expression(self, frame, objects, index):
        # apply TenCrop to image and convert each crop ToTensor
        transform_test = transforms.Compose([
            transforms.TenCrop(self.cut_size),
            transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
        ])
        # albumentations helper function
        def augment(aug, image):
            return aug(image=image)['image']
        # face bounding box coordinates as type: integer
        box_ints = objects[index]['box'].astype(int)
        # crop bounding box from frame, resize to 48x48, convert to Grayscale
        crop_aug = albu.Crop(box_ints[0], box_ints[1], box_ints[2], box_ints[3])
        img = augment(crop_aug, frame)
        size_aug = albu.Resize(48, 48)
        img = augment(size_aug, img)
        img = transforms.ToPILImage()(img)
        img = transforms.Grayscale()(img)
        # apply TenCrop to Resized + Grayscaled faces and convert ToTensor inputs
        inputs = transform_test(img)

        class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

        net = VGG('VGG19')
        checkpoint = torch.load('../FER/PrivateTest_model.t7',     # path to FER model
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

        print("The Expression is %s" %str(class_names[int(predicted.cpu().numpy())]))
        
    def __call__(self, frame: Image):
        objs = frame.objects
        for n in range(len(objs)):
            self.expression(frame, objs, n)

        return frame
