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
import argparse
from get_input_args import get_input_args
import json
    
  
    
def classify_model(model):
    #Classifications of models
    if (in_arg.arch=='vgg16'):
        model.classifier = nn.Sequential(OrderedDict([
                              ('fc1', nn.Linear(25088, 4096, bias=True)),
                              ('relu', nn.ReLU()),
                              ('dropout', nn.Dropout(p=0.5)),
                              ('fc2', nn.Linear(4096, 102, bias=True)),
                              ('output', nn.LogSoftmax(dim=1))
                              ]))
    
    elif (in_arg.arch=='densenet161'):
        model.classifier = nn.Sequential(OrderedDict([
                                  ('fc1', nn.Linear(1024, 500)),
                                  ('drop', nn.Dropout(p=0.6)),
                                  ('relu', nn.ReLU()),
                                  ('fc2', nn.Linear(500, 102)),
                                  ('output', nn.LogSoftmax(dim=1))]))

    
def train_model():
    print("-----------------Inside model training-------------------")
    epochs = in_arg.epochs
    train_losses, test_losses = [],[]
    current = 0

    for epoch in range(epochs):
        running_loss = 0
        for inputs, labels in trainloader:
            current += 1
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if current%40 == 0:
                test_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():

                    for images, labels in testloader:
                        images, labels = images.to(device), labels.to(device)
                        log_ps = model.forward(images)
                        test_loss += criterion(log_ps, labels)

                        ps = torch.exp(log_ps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor))

                model.train()

                print("Training loss: {:.4f}".format(running_loss/len(trainloader)),
                      "Test loss: {:.4f}".format(test_loss/len(testloader)),
                      "Test Accuracy: {:.4f}".format((accuracy/len(testloader))*100))

def validation():
    #Do validation on the test set
    valid_loss = 0
    valid_accuracy = 0
    model.eval()
    with torch.no_grad():
            for images, labels in validloader:
                        images, labels = images.to(device), labels.to(device)
                        log_ps = model.forward(images)
                        valid_loss += criterion(log_ps, labels)

                        ps = torch.exp(log_ps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        valid_accuracy += torch.mean(equals.type(torch.FloatTensor))

    print("Validation Accuracy: {:.4f}".format((valid_accuracy/len(validloader))*100))
    
    
def save_checkpoint(in_arg):
    #Save the checkpoint 
    print("----------------initialization saving process-------------------")
    if type(in_arg.save_dir)== type(None):
        print("Provide valid saving directory!!")
        return
    file_path = in_arg.save_dir + 'checkpoint.pth'
    model.class_to_idx = train_data.class_to_idx
    checkpointData = { 'arch':model.name,
                      'epochs':in_arg.epochs,
                      'dropout':0.5,
                      'classifier':model.classifier,
                      'class_to_idx':model.class_to_idx,
                      'state_dict': model.state_dict()}
    torch.save(checkpointData, file_path)
    


    #getting input arguments
in_arg = get_input_args()

print("### DEBUG: save_dir={}".format(in_arg.save_dir))
    
with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)

data_dir = in_arg.data_dir
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'
    
train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                          transforms.RandomResizedCrop(224),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

valid_transforms = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])


test_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])

    # TODO: Load the datasets with ImageFolder
train_data = datasets.ImageFolder(train_dir,transform=train_transforms)
valid_data = datasets.ImageFolder(valid_dir,transform=valid_transforms)
test_data = datasets.ImageFolder(test_dir,transform=test_transforms)

    # TODO: Using the image datasets and the trainforms, define the dataloaders
trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
validloader = torch.utils.data.DataLoader(valid_data, batch_size=64)
testloader = torch.utils.data.DataLoader(test_data, batch_size=64)

if torch.cuda.is_available() and in_arg.gpu == 'gpu':
    device = torch.device("cuda:0") 
else:
    device = torch.device("cpu")
    
if (in_arg.arch=='vgg16'):
        model= models.vgg16(pretrained=True)
        model.name = "vgg16"
        input_units = 25088    
elif (in_arg.arch=='densenet161'):
        model= models.densenet161(pretrained=True)
        model.name = "densenet161"
        input_units = 1024
else:
        print("Classifier model is invalid!!")


    
    #Freeze parameters
for param in model.parameters():
        param.requires_grad = False
    
classify_model(model)
    
criterion = nn.NLLLoss()

    #Only training the classifier parameters
optimizer = optim.Adam(model.classifier.parameters(), lr=in_arg.learning_rate)
    
model.to(device)
    
    #To train the model
train_model()
print("----------------- model training finished-------------------")
    #For validation of model
validation()
    
    #Used to save checkpoint
save_checkpoint(in_arg)
print("----------------finished saving process---------------------")









