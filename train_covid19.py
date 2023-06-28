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
import torchvision.models as models
import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import copy
import time
import pathlib 
from covidLUS.load_dataset import CovidUltrasoundDataset
from torchsummary import summary


# def set_parameter_requires_grad(model, feature_extracting):
#     if feature_extracting:
#         for param in model.parameters():
#             param.requires_grad = False


def train_model(model, dataloaders, criterion, optimizer, scheduler,epochs=3):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    # Initialize lists to store loss and accuracy values
    train_losses = []
    val_losses = []
    val_accuracies = []
    train_accuracies = []
    predicted_labels = []
    true_labels = []
    total = 0
    correct = 0 
    softmax_probs = []

    for epoch in range(epochs):
        print(f'Epoch {epoch}/{epochs - 1}')
        print('-' * 10)
        
        for phase in ['train', 'val']:

            if phase == 'train':
                model.train() # set model to training mode
            else:
                model.eval() # set modelt o evaluation mode
            
            running_loss = 0.0
            running_corrects = 0

            # Iterate over data 
            for inputs, labels in dataloaders[phase]:

                inputs, labels = inputs.to(device), labels.to(device)

                # zero te parameter gradients
                optimizer.zero_grad()
                
                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    softmax_output = F.softmax(outputs, dim=1)
                    softmax_probs.extend(softmax_output.detach().cpu().numpy())
                
                # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        
                
                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data).cpu()

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
            if phase == 'train':
                # Append the loss and accuracy values for each epoch
                scheduler.step()
                train_losses.append(epoch_loss)
                train_accuracies.append(epoch_acc)
            else:
                # Append the loss and accuracy values for validation set
                val_losses.append(epoch_loss)  
                val_accuracies.append(epoch_acc)

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            
        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    model.load_state_dict(best_model_wts)

    num_images = 5  # Number of images to visualize
    pdb.set_trace()

    # Get a random subset of images and corresponding softmax probabilities
    indices = np.random.choice(len(dataloaders['val'].dataset), num_images, replace=False)
    sample_images = [dataloaders['val'].dataset[i][0] for i in indices]
    sample_probs = [softmax_probs[i.item()] for i in indices if i.item() < len(softmax_probs)]    
    
    # testing loop
    model.eval()
    with torch.no_grad():
        for inputs, labels in dataloaders['val']:
            images, labels = inputs.to(device), labels.to(device)
            outputs = model(images)
            # outputs = model.forward(images)
            # batch_loss = criterion(outputs, labels)
            # ps = torch.exp(outputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += torch.sum(predicted == labels.data)
            predicted_labels.extend(predicted.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    print('Accuracy of the network on the 628 test images: %d %%' % (
        100 * correct / total))
    
    target_names = ['covid', 'pneumonia', 'regular'] # Add your own class names
    report = metrics.classification_report(true_labels, predicted_labels, target_names=target_names)
    print(report)

    # Plot the images and softmax probabilities
    plt.figure(figsize=(12, 8))
    for i in range(num_images):
        plt.subplot(2, num_images, i+1)
        plt.imshow(sample_images[i].permute(1, 2, 0))
        plt.axis('off')

        plt.subplot(2, num_images, num_images + i + 1)
        plt.bar(range(len(target_names)), sample_probs[i])
        plt.xticks(range(len(target_names)), target_names)
        plt.xlabel('Classes')
        plt.ylabel('Probability')

    plt.tight_layout()
    plt.show()

    # Plotting the graph
    plt.style.use('ggplot')
    plt.figure()
    plt.plot(range(1, epochs + 1), train_losses, label='Training Loss')
    plt.plot(range(1, epochs + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.title('Training Loss and Validation Loss')
    plt.legend(loc='lower left')
    plt.savefig('training_loss.png')

    plt.style.use('ggplot')
    plt.figure()
    plt.plot(range(1, epochs + 1), train_accuracies, label='Training Accuracy')
    plt.plot(range(1, epochs + 1), val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.title('Training Accuracy and Validation Accuracy')
    plt.legend(loc='lower left')
    plt.savefig('training_accuracy.png')
    
    return model

# data path variables 
data_path = pathlib.Path.cwd() / "data"
train_image_data_path = pathlib.Path(data_path) / f"new_train_dataset"
test_image_data_path = pathlib.Path(data_path) / f"new_test_dataset"
train_label_path = pathlib.Path(data_path) / f"new_train_annotation.csv"
test_label_path = pathlib.Path(data_path) / f"new_test_annotation.csv"

# set batch_size
batch_size = 8

# set number of workers
num_workers = 2

if __name__ == "__main__":

    # set device to cuda else cpu 
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # load train data
    train_covid_dataset = CovidUltrasoundDataset(
                                        annotations_file=train_label_path,
                                        img_dir=train_image_data_path,
                                        )
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
                                            img_dir=test_image_data_path,
                                            )
    print(f'len of test set: {len(test_covid_dataset)}')

    test_dataloader = DataLoader(
                            test_covid_dataset, 
                            batch_size=batch_size, 
                            shuffle=True,
                            num_workers=num_workers
                        )

    dataloaders = {
        "train": train_dataloader,
        "val": test_dataloader
    }
    model = models.vgg16(weights='IMAGENET1K_V1')

    for module in model.features.children():
        if isinstance(module, nn.Conv2d):
            module.add_module('BatchNorm', nn.BatchNorm2d(module.out_channels))

    # num_classes = 3
    # num_channels = model.classifier[6].in_features // (4 * 4)
    # in_features = num_channels * 4 * 4
    # model.avgpool = nn.AdaptiveAvgPool2d(output_size=(7, 7))
    # Modify the adaptive average pooling layer
    # model.classifier._modules['6'] = nn.AdaptiveAvgPool2d((4, 4))

    num_features = model.classifier[6].in_features
    # out_features = model.classifier[6].out_features

    # print(f'what is out_features{out_features}')

    # model.classifier[-1] = nn.Linear(in_features = num_features, 
    #                               out_features = 3,
    #                               bias = True)
    
    # for parameter in model.classifier[:-1].parameters():
    #     parameter.requires_grad = False
    # model.classifier[-1] = nn.LogSoftmax(dim=1)

    # model.classifier[-1] = nn.Sequential(
    #                     nn.Linear(in_features = num_features, 
    #                               out_features = 3,
    #                               bias = True),
    #                     nn.LogSoftmax(dim=1)
    #                     )   
    model.classifier[-1] = nn.Linear(in_features = 4096, 
                                     out_features = 3, 
                                     bias = True
                                    )

    print(f"********* Model Parameter *********")
    print(model)    

# model.classifier = nn.Sequential(
#         nn.Conv2d(512, 256, kernel_size=3, padding=1),
#         nn.ReLU(inplace=True),
#         nn.Conv2d(256, 128, kernel_size=3, padding=1),
#         nn.ReLU(inplace=True),
#         nn.Conv2d(128, 64, kernel_size=3, padding=1),
#         nn.ReLU(inplace=True),
#         nn.Conv2d(64, 3, kernel_size=1),
#         nn.ReLU(inplace=True),
#         nn.AdaptiveAvgPool2d((1,1))
#     )

# model.fc = nn.Sequential(nn.Linear(2048, 512),
#                              nn.ReLU(),
#                              nn.Dropout(0.2),
#                              nn.Linear(512, num_classes),
#                              nn.LogSoftmax(dim=1))

    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(model.parameters(), lr=0.001, momentum=0.9)
    #     optimizer = optim.Adam(model.parameters(), lr=0.001)

    # place the network into gpu
    model = model.to(device)

    # train_losses, test_losses, epoch_arr = [],[],[]
    train_accuracy_arr = []
    print_every = 100
    train_correct = 0 
    train_total = 0 
    steps = 0

    learning_rate = 1e-4
    # weight_decay =1e-4
    optimizer = optim.Adam(model.parameters(), 
                            lr=learning_rate,
                            )
    
    # optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    
    # Learning Rates

    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
    #                                             mode='min',
    #                                             factor=0.7, 
    #                                             patience=7, 
    #                                             # threshold=0.0001, 
    #                                             # threshold_mode='abs',
    #                                             eps=1e-4,
    #                                             verbose=True
    #                                         )
    scheduler = optim.lr_scheduler.StepLR(optimizer, 
                                          step_size=7, 
                                          gamma=0.1,
                                          verbose = True
                                          )
    model_ft = train_model(model = model, 
                           dataloaders = dataloaders, 
                           criterion = criterion, 
                           optimizer = optimizer, 
                           scheduler = scheduler, 
                           epochs=2)
