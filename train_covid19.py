# import matplotlib.pyplot as plt # for plotting
# import numpy as np # for transformation
import pdb

import torch # PyTorch package
import torchvision # load datasets
import torchvision.transforms as transforms # transform data
import torch.nn as nn # basic building block for neural neteorks
import torch.nn.functional as F # import convolution functions like Relu
import torch.optim as optim # optimzer
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from sklearn import metrics
from torchvision import models


import pathlib 
from covidLUS.load_dataset import CovidUltrasoundDataset

# VGG16 model 
class VGG16(nn.Module):
    def __init__(self, num_classes=3):
        super(VGG16, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU())
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU())
        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.layer5 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU())
        self.layer6 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU())
        self.layer7 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.layer8 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.layer9 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.layer10 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.layer11 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.layer12 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.layer13 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(7*7*512, 4096),
            nn.ReLU())
        self.fc1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU())
        self.fc2= nn.Sequential(
            nn.Linear(4096, num_classes))
        
    def forward(self, x):
        if x.shape[1]== 1:
            x = x.squeeze(1)
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)
        out = self.layer8(out)
        out = self.layer9(out)
        out = self.layer10(out)
        out = self.layer11(out)
        out = self.layer12(out)
        out = self.layer13(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out

data_path = pathlib.Path.cwd() / "data"
train_image_data_path = pathlib.Path(data_path) / f"train_dataset"
test_image_data_path = pathlib.Path(data_path) / f"test_dataset"
train_label_path = pathlib.Path(data_path) / f"train_annotation.csv"
test_label_path = pathlib.Path(data_path) / f"test_annotation.csv"

# transform = transforms.Compose( # composing several transforms together
#     [transforms.ToTensor(), # to tensor object
#      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) # mean = 0.5, std = 0.5
# transform = transforms.Compose([
#                     transforms.Grayscale(num_output_channels=3),
#                     transforms.Resize(image_size),
#                     transforms.RandomHorizontalFlip(),
#                     transforms.ToTensor(),
#                     # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.1),
#                     # transforms.RandomAffine(degrees=40, translate=None, scale=(1, 2), shear=15, interpolation=False, fill=0),
#                     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
#                     # transforms.Lambda(lambda x: x[0]) # remove color channel
#                 ])

# set batch_size
batch_size = 8

# set number of workers
num_workers = 2

if __name__ == "__main__":
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # pdb.set_trace()
    # load train data
    train_covid_dataset = CovidUltrasoundDataset(
                                        annotations_file=train_label_path,
                                        img_dir=train_image_data_path,
                                        )
    # print(f'len of train set of org_{i+1}: {len(train_covid_dataset)}')
    print(f'len of train set: {len(train_covid_dataset)}')

    train_dataloader = DataLoader(
                                train_covid_dataset, 
                                batch_size=batch_size, 
                                shuffle=True,
                                num_workers=num_workers,
                                )
                                

    # load test data 
    test_covid_dataset = CovidUltrasoundDataset(
                                            annotations_file=test_label_path,
                                            img_dir=test_image_data_path
                                            )
    # print(f'len of test set of org_{i+1}: {len(test_covid_dataset)}')
    print(f'len of test set: {len(test_covid_dataset)}')

    test_dataloader = DataLoader(
                            test_covid_dataset, 
                            batch_size=batch_size, 
                            shuffle=False,
                            num_workers=num_workers
                        )
    # iter the train dataset
    train_features, train_labels = next(iter(train_dataloader))

#     model = VGG16(3).to(device)
    model = models.vgg16(pretrained=True)
    
    print(f"********* Model Parameter *********")
    print(model)
    
    for param in model.parameters():
        param.requires_grad = False
    
    model.fc = nn.Sequential(nn.Linear(2048, 512),
                                 nn.ReLU(),
                                 nn.Dropout(0.2),
                                 nn.Linear(512, 10),
                                 nn.LogSoftmax(dim=1))
    
    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(model.parameters(), lr=0.001, momentum=0.9)
#     optimizer = optim.Adam(model.parameters(), lr=0.001)
    optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

    model.to(device)
    # pdb.set_trace()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    epochs = 1
    running_loss = 0.0
    train_losses, test_losses = [],[]
    print_every = 100
    steps = 0
    
    start.record()
    
    for epoch in range(epochs):  # loop over the dataset multiple times
#         print(f'training iter {epoch}')
        for i, (inputs, labels) in enumerate(train_dataloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            # inputs, labels = data
            inputs, labels = inputs.cuda(), labels.cuda()
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = model(inputs)
#             outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            
            if steps % print_every == 0:
                test_loss = 0
                accuracy = 0
                model.eval()
                train_losses.append(running_loss/len(train_dataloader))
                print(f"Epoch {epoch+1}/{epochs}.. "
                   f"Train loss: {running_loss/print_every:.3f}.. ")
                running_loss = 0.0
                model.train()
                
                 
#                       f"Test loss: {test_loss/len(test_dataloader):.3f}.. "
#                       f"Test accuracy: {accuracy/len(test_dataloader):.3f}")
                  
                
            
#             if i % 100 == 99:    # print every 2000 mini-batches
#                 print('[%d, %5d] loss: %.3f' %
#                     (epoch + 1, i + 1, running_loss / 100))
#                 running_loss = 0.0
    end.record()

    # Waits for everything to finish running
    torch.cuda.synchronize()

    print('Finished Training')
    print(start.elapsed_time(end))

    correct = 0
    total = 0
    
    test_loss = 0 
    accuracy = 0
    
    with torch.no_grad():
        for (inputs, labels) in test_dataloader:
            # pdb.set_trace()
            images, labels = inputs.cuda(), labels.cuda()
            outputs = model(images)
#             outputs = model.forward(images)
            batch_loss = criterion(outputs, labels)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            test_loss += batch_loss.item()
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
            test_losses.append(test_loss/len(test_dataloader))
            print(f"Test loss: {test_loss/len(test_dataloader):.3f}.."
                  f"Test accuracy: {accuracy/len(test_dataloader):.3f}")
    # Accuracy = metrics.accuracy_score(total, correct)

    # print(f'what is accuracy: {Accuracy}')

    print('Accuracy of the network on the 523 test images: %d %%' % (
        100 * correct / total))
