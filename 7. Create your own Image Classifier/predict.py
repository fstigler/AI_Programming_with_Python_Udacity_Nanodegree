import matplotlib.pyplot as plt

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict
import PIL
from PIL import Image
import numpy as np
import seaborn as sns
from math import ceil
import argparse
from get_input_args_pred import get_input_args_pred
import json
import os

in_arg = get_input_args_pred()

print("######")
print(in_arg)

with open(in_arg.category_names, 'r') as f:
        cat_to_name = json.load(f)

checkpointData = torch.load(in_arg.checkpoint)

if checkpointData ['arch'] == 'vgg16':
    model = models.vgg16(pretrained=True)
    model.name = "vgg16"
else: 
    model = models.checkpointData['arch'](pretrained=True)
    model.name = checkpointData['arch']
    
for param in model.parameters(): 
    param.requires_grad = False
    
def model_details(checkpointData):
    model.classifier = checkpointData['classifier']

    model.class_to_idx = checkpointData['class_to_idx']

    model.load_state_dict(checkpointData['state_dict'])
    
    return model
    
model = model_details(checkpointData)


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    img = Image.open(image)
    
    if img.size[0] > img.size[1]:
        img.thumbnail((30000, 256))
    else:
        img.thumbnail((256, 30000))
    
    img = img.crop(((img.width-224)/2,
                    (img.height-224)/2,
                    (img.width-224)/2 + 224,
                    (img.height-224)/2 + 224))
    
    img=np.array(img)/255
    
    img= (img-np.array([0.485, 0.456, 0.406]))/(np.array([0.229, 0.224, 0.225]))
    
    img= img.transpose((2,0,1))
    
    return img
   

def predict(image_path, model, topk=in_arg.top_k):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    if torch.cuda.is_available():
        model.cuda()
    else:
        model.to(device)
    
    img = process_image(image_path)
    image_tensor = torch.from_numpy(img).type(torch.FloatTensor).to("cpu")
    inputs = image_tensor.unsqueeze(0)
    output = model.forward(inputs)
    pb= torch.exp(output)
    
    top_pb, top_class = pb.topk(topk)
    top_pb = top_pb.tolist()[0]
    top_class = top_class.tolist()[0]
    
    

    data = {val: key for key, val in model.class_to_idx.items()}
    top_flow = []
    
    for i in top_class:
        i_ = "{}".format(data.get(i))
        top_flow.append(cat_to_name.get(i_))

    return top_pb, top_flow

probability,flower = predict(os.path.join(in_arg.image,in_arg.image_name), model)

no_of_pred = in_arg.top_k

for i in range(no_of_pred):
    print("For category - {} ==>  percentage - {} %".format(flower[i],ceil(probability[i]*100)))
