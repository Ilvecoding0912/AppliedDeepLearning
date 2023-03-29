import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
import time
import torchvision.transforms as transforms
import torchvision.models as models
import os, sys
from data import *

class ModelResNet50(nn.Module):
    '''
    Create a ResNet50 model and change the last fully connected layer for ResNet50 model in pytorch
    The out_features is equal to the number of classes
    '''
    def __init__(self, out_features):
        super(ModelResNet50, self).__init__()

        self.model = models.resnet50(weights=True)
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, out_features)
    
    def forward(self, x):
        output = self.model(x)
        return output
    

class Ablation:
    '''
    Training and testing the model, compare its performance with different optimzers and sampling methods

    Args:
    model: resnet50
    device: cuda or cpu
    optimizer: sgd or adam
    sampling method: 1 => beta distribution     2 => uniform sampling
    '''
    def __init__(self, model, device, optimizer='sgd', sampling_method=1):
        super(Ablation).__init__()
        assert optimizer in ['sgd', 'adam'], "Optimizer can only be 'sgd' or 'adam'"
        assert sampling_method in [1, 2], "Sampling method can only be 1 or 2. 1 => Beta sampling, 2 => Uniformly sampling"

        self.model = model.to(device)
        self.device = device
        self.sampling_method = sampling_method
        self.data_augmentation = MixUp(sampling_method=sampling_method, device=device)
        self.criterion = nn.CrossEntropyLoss()

        if(optimizer == 'sgd'):
            self.optimizer_choice = optimizer
            self.optimizer = optim.SGD(self.model.parameters() ,lr=0.001, momentum=0.9)
        else:
            self.optimizer_choice = optimizer
            self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
    
    def run(self, trainloader, valloader, num_epochs=10):
        '''
        Training, testing and saving the resnet model with data augmentation. 
        
        Args:
        trainloader: dataloader for training dataset
        valloader: dataloader for validation dataset
        num_epochs: number of epochs
        '''
        print('\n')
        print('==> Starting training and validation...')
        print(f'Optimiser: {self.optimizer_choice} \t Sampling method: {self.sampling_method}')

        device = self.device

        train_time, val_time = 0, 0
        for epoch in range(num_epochs):
            train_loss, val_loss = 0, 0
            train_correct, val_correct = 0, 0
        
            # Training process
            self.model.train()
            train_start_time = time.time()
            for data, labels in trainloader:
                data = data.permute(0, 3, 1, 2).float()
                
                data, labels = data.to(device), labels.to(device)
         
                mixed_data, y_a, y_b, lam = self.data_augmentation.mixup_data(data, labels, 0.2)
                outputs = self.model(mixed_data)
                
                # Training accuracy
                pred_labels = torch.argmax(outputs, dim=1)
                train_correct += lam * (y_a == pred_labels).sum().item() + (1 - lam) * (y_b == pred_labels).sum().item()
                
                # Training loss
                loss = self.data_augmentation.mixup_loss(outputs, y_a, y_b, lam)
                train_loss += loss.item() * data.size(0)
           
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            train_end_time = time.time()
            train_time += (train_end_time - train_start_time)

            # Validation process
            self.model.eval()
            val_start_time = time.time()
            for data, labels in valloader:
                with torch.no_grad():
                    data = data.permute(0, 3, 1, 2).float()
                    data, labels = data.to(device), labels.to(device)
                    outputs = self.model(data)

                    # Validation accuracy
                    pred_labels = torch.argmax(outputs, dim=1)
                    val_correct += (labels == pred_labels).sum().item()
                    
                    # Validation loss
                    loss = self.criterion(outputs, labels)
                    val_loss += loss.item() * data.size(0)

            val_end_time = time.time()
            val_time += (val_end_time - val_start_time)

            self.train_loss, self.val_loss = train_loss / len(trainloader.dataset), val_loss / len(valloader.dataset)
            self.train_accuracy, self.val_accuracy = train_correct / len(trainloader.dataset), val_correct / len(valloader.dataset)

            print(f'Epoch: {epoch+1}\t Training Loss: {self.train_loss}\t Training Acc: {self.train_accuracy}\t Validation Loss: {self.val_loss}\t Validation Accuracy: {self.val_accuracy}')
        
        print('==> Training and validation end...')

        self.train_speed = train_time / num_epochs
        self.val_speed = val_time / num_epochs

        # Summary
        print('\n')
        print('------------------------------------------Training & Validation Summary-------------------------------------------------')
        print(f'|\t\t\t Sampling method: {self.sampling_method}\t\t\t  Optimizer: {self.optimizer_choice} \t\t\t\t\t |')
        print('------------------------------------------------------------------------------------------------------------------------')
        print('|\t Training Loss: %.4f\t |\t Training Accuracy: %.3f\t | Average Training Speed (per epoch): %.3f |' % (self.train_loss, self.train_accuracy, self.train_speed))
        print('------------------------------------------------------------------------------------------------------------------------')
        print('|\t Validation Loss: %.4f |\t Validation Accuracy: %.3f\t | Average Validation Speed (per epoch): %.3f |' % (self.val_loss, self.val_accuracy, self.val_speed))
        print('--------------------------------------------------------------------------------------------------------------------------')
        print('\n')

        # Save the model after training
        if self.optimizer_choice == 'sgd':
            # TODO Users can find the model under 'models' folder
            torch.save(self.model, './models/resnet50_sgd.pth')
            print("Model with sgd optimizer is saved.")
        
        elif self.optimizer_choice == 'adam':
            # TODO Users can find the model under 'models' folder
            torch.save(self.model, './models/resnet50_adam.pth')
            print("Model with adam optimizer is saved.")

    def test(self, testloader):
        '''
        Run the model on testing dataset

        Args:
        testloader: dataloader for testing dataset
        '''
        print('==> Starting testing...')
        print(f'Optimiser: {self.optimizer_choice} \t Sampling method: {self.sampling_method}')

        device = self.device

        # TODO Users can change the path to their directory of models
        if os.path.exists('./models/resnet50_sgd.pth') and self.optimizer_choice == 'sgd':
            print('ResNet50 sgd model exits')
            self.model = torch.load('./models/resnet50_sgd.pth', map_location=torch.device(device))
        elif os.path.exists('./models/resnet50_adam.pth') and self.optimizer_choice == 'adam':
            print('ResNet50 adam model exits')
            self.model = torch.load('./models/resnet50_adam.pth', map_location=torch.device(device))
        
        self.model.eval()
        test_correct, test_loss = 0, 0
        test_time = 0
        for data, labels in testloader:
            with torch.no_grad():
                test_start_time = time.time()
                data = data.permute(0, 3, 1, 2).float()
                data, labels = data.to(device), labels.to(device)
                outputs = self.model(data)

                # Test accuracy
                pred_labels = torch.argmax(outputs, dim=1)
                test_correct += (labels == pred_labels).sum().item()

                # Test loss
                loss = self.criterion(outputs, labels)
                test_loss += loss.item() * data.size(0)

                test_end_time = time.time()
                test_time += (test_end_time - test_start_time)

        self.test_loss = test_loss / len(testloader.dataset)
        self.test_accuracy = test_correct / len(testloader.dataset)
        self.test_speed = test_time
        print('\n')
        print('----------------------------Testing Summary-----------------------------')
        print(f'|\t Sampling method: {self.sampling_method}\t\t  Optimizer: {self.optimizer_choice}\t\t|')
        print('------------------------------------------------------------------------')
        print('|\t Testing Loss: %.4f\t |\t Testing Accuracy: %.3f \t |' % (self.test_loss, self.test_accuracy))
        print('------------------------------------------------------------------------')
        print('==> Testing end...')
        print('\n')

class CustomSubset(Dataset):
    """
    Subset of a dataset at specified indices.
    """
    def __init__(self, dataset, indices):
        self.dataset = torch.utils.data.Subset(dataset, indices)
        self.data = self.dataset[:][0]
        self.targets = self.dataset[:][1]

    def __getitem__(self, idx):
        image = self.dataset[idx][0]
        target = self.targets[idx]
        return (image, target)

    def __len__(self):
        return len(self.targets)
    
class CustomConcatDataset(Dataset):
    '''
    Concat two datasets wrt their data and targets
    '''
    def __init__(self, dataset1, dataset2):
        self.dataset = torch.utils.data.ConcatDataset([dataset1, dataset2])
        self.data = torch.cat((torch.tensor(dataset1.data), torch.tensor(dataset2.data)))
        self.targets = torch.cat((torch.tensor(dataset1.targets), torch.tensor(dataset2.targets)))
    
    def __getitem__(self, idx):
        image = self.data[idx]
        target = self.targets[idx]
        return (image, target)
    
    def __len__(self):
        return len(self.targets)


def split_dataset(dataset, test_size=0.2):
    '''
    Split the dataset with a split size of test_size

    Args:
    dataset: dataset that will be splitted
    test_size: a splitting ratio
    '''
    indices = torch.randperm(len(dataset))
    train_indices, test_indices = indices[:int((1-test_size) * len(indices))], indices[int((1-test_size) * len(indices)):]

    train_dataset = CustomSubset(dataset, train_indices)
    test_dataset = CustomSubset(dataset, test_indices)

    return train_dataset, test_dataset

if __name__ == '__main__':

    transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    
    merged_dataset = CustomConcatDataset(train_dataset, test_dataset)

    # Split CIFAR-10 dataset into development set (80%) and holdout test set (20%)
    devset, testset = split_dataset(merged_dataset, 0.2)

    # Random-split the development set in the train (90%) and validation sets (10%)
    trainset, valset = split_dataset(devset, 0.1)

    trainloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=0)
    valloader = DataLoader(valset, batch_size=128, shuffle=False, num_workers=0)
    testloader = DataLoader(testset, batch_size=128, shuffle=False, num_workers=0)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Current device: {device}\n')

    # Initialize resnet50 model
    out_features = len(np.unique(trainset.targets))
    resnet50 = ModelResNet50(out_features)

    print('\n')
    print('Compare the results with SGD and Adam optimizers and both sampling methods are 1\n')
    
    # Initialize the class with different optimizers and the same sampling method (mixup)
    ab_sgd = Ablation(resnet50, device, 'sgd', 1)
    ab_adam = Ablation(resnet50, device, 'adam', 1)

    # Training and Validation process
    ab_sgd.run(trainloader, valloader)
    ab_adam.run(trainloader, valloader)
    
    # Testing the models
    ab_sgd.test(testloader)
    ab_adam.test(testloader)

    print('Comparison of performance on test set and development set:')
    print('Taking SGD optimiser for example: The accuracy and loss of the model on the test dataset and the validation dataset are similar, \n while accuracy is slightly higher on training dataset, which is about 86%.')
    print('Training process takes much long time than testing and validation process per epoch.')
    print('\n')
    print('Comparison of SGD and Adam optimisers:')
    print('In general, model with SGD optimsers have better performance on loss and accuracy value on all the dataset.\n But the learning rate of these two optimizers are 0.001, which may be not the appropriate parameters for adam optimiser.')
    print('\n')



