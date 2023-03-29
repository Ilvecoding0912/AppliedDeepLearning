import numpy as np
import torch
import torch.nn as nn
import torchvision
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
import time
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import os, sys
from data import *

class ModelResNet50(nn.Module):
    '''
    Create a ResNet50 model and change the last fully connected layer for ResNet50 model in pytorch
    The out_features is equal to the number of classes

    Args:
    out_features: the number of classes for the current dataset

    Returns:
    output: output value from current model
    '''
    def __init__(self, out_features):
        super(ModelResNet50, self).__init__()

        self.model = models.resnet50(weights=True)
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, out_features)
    
    def forward(self, x):
        output = self.model(x)
        return output

class ImageClassification:
    '''
    Define the training and testing process for a given model and a sampling method

    Args:
    model: a machine learning model for image classification
    device: 'cuda' or 'cpu'
    sampling_method: 1 or 2 corresponding to different sampling methods (beta or uniform)
    '''

    def __init__(self, model, device, sampling_method=1):
        self.device = device
        self.model = model.to(device)
        self.sampling_method = sampling_method
        self.optimizer = optim.SGD(self.model.parameters() ,lr=0.001, momentum=0.9)
        self.data_augmentation = MixUp(sampling_method=sampling_method, device=device)

    def run(self, trainloader, testloader, num_epochs=10, save_montages=True):
        '''
        Training, testing and saving the resnet model with data augmentation. 
        If the model exits in the given folder, then the exiting model will be used.

        Args:
        trainloader: dataloader for training dataset
        testloader: dataloader for testing dataset
        num_epochs: number of training and testing epochs
        save_montages: bool variable, True => save montages of mixup data   False => don't save them
     
        '''
        print('\n')
        print('==> Starting training and testing...')
        device = self.device

        # TODO Users can directly load the models which are under 'models' folder
        # if os.path.exists('./models/resnet50_mixup.pth') and self.sampling_method == 1:
        #     print('ResNet50 mixup model exits')
        #     self.model = torch.load('./models/resnet50_mixup.pth', map_location=torch.device(device))
        # elif os.path.exists('./models/resnet50_uniform.pth') and self.sampling_method == 2:
        #     print('ResNet50 uniform model exits')
        #     self.model = torch.load('./models/resnet50_uniform.pth', map_location=torch.device(device))


        save_train_img = True
        for epoch in range(num_epochs):
            train_loss = 0
            correct = 0
            # Training process
            self.model.train()
            for data, labels in trainloader:
                data, labels = data.to(device), labels.to(device)
        
                mixed_data, y_a, y_b, lam = self.data_augmentation.mixup_data(data, labels, 0.2)
                outputs = self.model(mixed_data)
                
                # Save 16 images as a montage
                if(save_train_img and save_montages):
                    save_images(mixed_data[:16], mode='train')
                    save_train_img = False

                # Training loss
                loss = self.data_augmentation.mixup_loss(outputs, y_a, y_b, lam)
                train_loss += loss.item() * data.size(0)
           
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            # Testing process
            self.model.eval()
            for data, labels in testloader:
                with torch.no_grad():
                    data, labels = data.to(device), labels.to(device)
                    outputs = self.model(data)

                    pred_labels = torch.argmax(outputs, dim=1)
                    correct += (labels == pred_labels).sum().item()
                    
            print(f'Epoch: {epoch+1} \t Training Loss: {train_loss / len(trainloader.dataset)} \t Testing Accuracy: {correct / len(testloader.dataset)}')

            # Save 36 test images as a montage
            if(save_montages):
                # Obtain the data
                dataiter = iter(testloader)
                images, labels = next(dataiter)

                # Predict the results
                with torch.no_grad():
                    self.model.eval()
                    outputs = self.model(images.to(device))
                    pred_labels = torch.argmax(outputs, dim=1)

                # Save images and print labels
                save_images(images[:36], 'test', labels[:36], pred_labels[:36])

        print('==> Training and Testing end...')
        print('\n')

        # Save the model to the 'models' folder
        if self.sampling_method == 1:
            torch.save(self.model, './models/resnet50_mixup.pth')
            print("Mixup model saved.")
        
        elif self.sampling_method == 2:
            torch.save(self.model, './models/resnet50_uniform.pth')
            print("Uniformly sampled model saved.")
    
if __name__ == '__main__':

    transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    # Download dataset 
    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    # Generate dataloaders
    trainloader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
    testloader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)

    # Initialize the resnet50 model
    out_features = len(np.unique(train_dataset.targets))
    resnet50 = ModelResNet50(out_features)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Current device: {device}')

    # Create different instances for different sampling methods
    task_mixup = ImageClassification(resnet50, device, sampling_method=1)
    task_uniform = ImageClassification(resnet50, device, sampling_method=2)

    # Sampling method 1 => beta sampling
    task_mixup.run(trainloader, testloader)

    # Sampling method 2 => uniform sampling
    task_uniform.run(trainloader, testloader)



    