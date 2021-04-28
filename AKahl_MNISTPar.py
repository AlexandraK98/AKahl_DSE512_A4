#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Training Data with Parallelism
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms
from tqdm import tqdm
from torchvision.models import densenet121
import torch.nn as nn

#adding data parallelism
class DataParallelModel(nn.Module):
#defing a model class
#class VolModel(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.num_classes = num_classes
        
        self.conv1 = nn.Conv2d(1, 4, 3)
        self.conv1 = nn.DataParallel(self.conv1)
        self.activation = nn.ReLU()
        self.conv2 = nn.Conv2d(4, 10, 3)
        self.conv2 = nn.DataParallel(self.conv2)
        self.pool = nn.AdaptiveMaxPool2d((1,1))
        self.classifier = nn.Linear(10, num_classes)
    def forward(self, x):
        x = self.conv1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.activation(x)
        x = self.pool(x)
        x = x.squeeze(-1).squeeze(-1)
        x = self.classifier(x)
        #nn.Sequential(nn.Conv2d(1, 4, 3), nn.ReLU())
        return x
        
#define a main training function
def train(num_epochs=30, batch_size=128):
        #set up MNIST dataset
        train_dataset = MNIST(root = '.', train=True, transform=transforms.Compose([transforms.ToTensor(),]), download=True)
        test_dataset = MNIST(root = '.', train = False, download= True)
        print(len(train_dataset), len(test_dataset))
        
        train_dataloaders = DataLoader(train_dataset, batch_size=batch_size)
        
        im, label = train_dataset[0]
        print(im.shape, label)
        
        
        #instantiate a model from our class
        model = DataParallelModel(num_classes = 10)
        #set up optimizer
        optimizer = optim.SGD(model.parameters(), lr=1e-2)
        #set up a loss criterion
        criterion = nn.CrossEntropyLoss()
        #loop for some number of epochs
        iter_losses = []
        epoch_losses = []
        for ep in range(num_epochs):
            ep_loss = 0
            for X, Y in tqdm(train_dataloaders):
                print(Y)
                #break
                #loop again over the dataset (minibatches)
                        #zero the gradients (resets the optimizer)
                optimizer.zero_grad()
                        # evaluate model on minibatch to get predictions
                pred = model(X)
                        # compare predictions to labels to get loss
                loss = criterion(pred, Y)
                        #record the loss
                iter_losses.append(loss.item())
                ep_loss += loss.item()
                        # compute gradient to backpropagation from loss
                loss.backward()
                        # step the optimizer
                optimizer.step()
            print(ep_loss)
            epoch_losses.append(ep_loss)
                # print the loss for this epoch
            return iter_losses, epoch_losses
if __name__ == '__main__':
        train()

