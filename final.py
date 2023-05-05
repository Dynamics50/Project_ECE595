import os
import glob
from torchvision import transforms
import numpy as np
import pandas as pd
import torch
import torchvision
import collections, random
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data import DataLoader
import random
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn.model_selection import train_test_split
import cv2 
import torch
import torch.optim as optim


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class DataGenerator():
    def __init__(self, n):
        self.train_data_path="C:/Users/md4to/Downloads/hw4_data/src/implementdata/train"
        self.train_image_paths= glob.glob(os.path.join(self.train_data_path,"*.jpg"))
        self.data_size = len(self.train_image_paths) if n is None else n
     
    def load_data(self):
        data = []
        for img_path in tqdm(self.train_image_paths):                        #tqdm for the progress time
            image_locator = os.path.splitext(os.path.basename(img_path))[0]
            img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)      #cv2.cvtColor converts the BGR to RGB
            label_img_id=image_locator.split("_")
            if len(label_img_id) == 1:
                label = 0 if "cat" in image_locator else 1
                img_id = label_img_id[0]
            else:
                label, img_id = label_img_id
                label = 0 if "cat" in label else 1
            data.append((label,img))
        data = data[:15000]
        random.shuffle(data)                       # shuffling
        print("Noise free Images")
        print(img.shape)
        return data
    
    def add_noise(self, data, po1, po2):           # po1=noise rate of positive label
     noise_data = [list(j) for j in data]          # po2=noise rate of negative label
     po1, po2 = float(po1), float(po2)
     swap1, swap_1 = noise_data.count(1)*po2, noise_data.count(-1)*po1
     for j in noise_data:
        if swap1 != 0 and j[0] == 1:              # Swapping the labels to generate noise into images
            if random.random()<swap1:
                j[0]= -1
                swap1-= 1
        elif swap_1!=0 and j[0]== -1:
            if random.random()<swap_1:
                j[0]=1
                swap_1-=1
     print("Noised Images:")
     return noise_data

    def dividedtt_data(self,data):                 #spliting
        train, test = train_test_split(data, train_size=0.8, shuffle=True)
        print ("Splitted Train & Test:")
        train,test = np.array(train),np.array(test)
        print("Train Size:",len(train))
        print("Test Size:",len(test))
        return train, test
    
    def generate_noised_data(self):
        data = self.load_data()
        noised_data = self.add_noise(data, 0.40, 0.30)
        noised_train_data, noised_test_data = self.dividedtt_data(noised_data)
        return noised_train_data, noised_test_data

def main_part():
    generate_data=DataGenerator(n=None)
    data = generate_data.load_data()
    noise_free_train_data, noise_free_test_data = generate_data.dividedtt_data(data)
    noised_data = generate_data.add_noise(data, 0.40, 0.30)
    noised_train_data, noised_test_data = generate_data.dividedtt_data(noised_data)
    noised_train_data, noised_test_data = generate_data.generate_noised_data()

    train_transforms = transforms.Compose([transforms.ToPILImage(),transforms.Resize((224, 224)),transforms.ToTensor(),transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    test_transforms = transforms.Compose([transforms.ToPILImage(),transforms.Resize((224, 224)),transforms.ToTensor(),transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    
    train_data = [(train_transforms(d[1]), d[0]) for d in noised_train_data]
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

    test_data = [(test_transforms(d[1]), d[0]) for d in noised_test_data]
    test_loader = DataLoader(test_data, batch_size=32, shuffle=True)

    # VGG16 model Loaded
    #vgg19_model = torchvision.models.VGG16_Weights(pretrained=True)
    vgg19_model = torchvision.models.vgg19(pretrained=True)

    # Layers 
    for feature in vgg19_model.parameters():
        feature.requires_grad = False
    num_features = vgg19_model.classifier[6].in_features
    classifier = nn.Sequential(nn.Linear(num_features, 512),nn.ReLU(inplace=True),nn.Linear(512, 2),nn.Softmax(dim=1))
    vgg19_model.classifier[6] = classifier
    model = vgg19_model.to(device)
    criterion = nn.CrossEntropyLoss()                      #Cross entropy loss
    #optimizer = optim.RMSprop(model.parameters(),lr=1e-4)
    #optimizer = optim.Adam(model.parameters(), lr=1e-4)
    optimizer = optim.SGD(model.parameters(), lr=1e-4)

    for epoch in range(10):                                #Model Training
        running_loss=0
        progress_bar=tqdm(train_loader, desc=f'Training Epoch {epoch + 1}/{10}', leave=False)
        for images, targets in progress_bar:
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()*images.size(0)
        epoch_loss = running_loss/len(train_loader.dataset)
        print(f'Training Loss: {epoch_loss:.4f}')
        progress_bar.close()
    
    with torch.no_grad():
     truth,total=0,0
     for images, labels in test_loader:
        images, labels = images.to(device),labels.to(device)
        outputs = vgg19_model(images)
        _, predicted = torch.max(outputs.data, 1)
        total+= labels.size(0)
        truth+= (predicted == labels).sum().item()
    accuracy = 100*truth/total
    print('Validation accuracy: {:.2f}%'.format(accuracy))
    vgg19_model.eval()                                      # Evaluation of model on test_Data
    test_loss = 0
    test_acc = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)  # images.to(device) for CPU
            outputs = vgg19_model(images)
            loss = criterion(outputs, labels)   # criterion for the loss fucntion between predicted and actual labels
            test_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            test_acc += torch.sum(preds == labels.data)
    test_loss/= len(test_loader)
    test_acc = test_acc.double()/len(test_data)
    print('Loss in Test: %.3f, Test Accuracy: %.3f'%(test_loss,test_acc))

if __name__ == '__main__':
    main_part()