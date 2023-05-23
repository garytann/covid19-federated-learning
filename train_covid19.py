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
from sklearn.preprocessing import LabelBinarizer
import copy
import time
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

def train_model(model, dataloaders, criterion, optimizer, scheduler,epochs=3):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    # Initialize lists to store loss and accuracy values
    train_losses = []
    train_accuracies = []
    predicted_labels = []
    true_labels = []
    total = 0
    correct = 0 


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
                    
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data).sum().cpu()
            
            if phase == 'train':
                scheduler.step()
            
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            # epoch_acc = running_corrects / len(dataloaders[phase].dataset)

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
            if phase == 'train':
                # Append the loss and accuracy values for each epoch
                train_losses.append(epoch_loss)
                train_accuracies.append(epoch_acc)

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    model.load_state_dict(best_model_wts)

    # testing loop
    model.eval()
    with torch.no_grad():
        for (inputs, labels) in dataloaders['val']:
            
            images, labels = inputs.cuda(), labels.cuda()
            outputs = model(images)
            # outputs = model.forward(images)
            
            # batch_loss = criterion(outputs, labels)
            # ps = torch.exp(outputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            # correct += (predicted == labels).sum().item()
            correct += torch.sum(predicted == labels.data)
            predicted_labels.extend(predicted.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    
            # top_p, top_class = ps.topk(1, dim=1)  
            # equals = top_class == labels.view(*top_class.shape)
            # test_loss += batch_loss.item()
            # accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
            # test_losses.append(test_loss/len(test_dataloader))
            # print(f"Test loss: {test_loss/len(test_dataloader):.3f}.."
            #       f"Test accuracy: {accuracy/len(test_dataloader):.3f}")
    # Accuracy = metrics.accuracy_score(total, correct)

    # print(f'what is accuracy: {Accuracy}')

    print('Accuracy of the network on the 523 test images: %d %%' % (
        100 * correct / total))
    
    target_names = ['0', '1', '2'] # Add your own class names
    report = metrics.classification_report(true_labels, predicted_labels, target_names=target_names)
    print(report)

    # Plotting the graph
    plt.style.use('ggplot')
    plt.figure()
    plt.plot(range(1, epochs + 1), train_losses, label='Training Loss')
    plt.plot(range(1, epochs + 1), train_accuracies, label='Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.title('Training Loss and Accuracy')
    plt.legend(loc='lower left')
    plt.savefig('training_loss.png')
    
    return model

# data path variables 
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
batch_size = 16

# set number of workers
num_workers = 2

if __name__ == "__main__":

    # set device to cuda else cpu 
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    train_img_labels = pd.read_csv(train_label_path)
    train_label = train_img_labels['label']
    test_img_labels = pd.read_csv(test_label_path)
    test_label = test_img_labels['label']

    lb = LabelBinarizer()
    train_labels = lb.fit(train_label)
    test_labels = lb.fit(test_label)
    # load train data
    train_covid_dataset = CovidUltrasoundDataset(
                                        annotations_file=train_label_path,
                                        img_dir=train_image_data_path,
                                        label = train_label
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
                                            img_dir=test_image_data_path,
                                            label = test_label
                                            )
    # print(f'len of test set of org_{i+1}: {len(test_covid_dataset)}')
    print(f'len of test set: {len(test_covid_dataset)}')

    test_dataloader = DataLoader(
                            test_covid_dataset, 
                            batch_size=batch_size, 
                            shuffle=False,
                            num_workers=num_workers
                        )

    dataloaders = {
        "train": train_dataloader,
        "val": test_dataloader
    }
    # iter the train dataset
    train_features, train_labels = next(iter(dataloaders['train']))
    #     model = VGG16(3).to(device)
    # model = models.vgg16(pretrained=True)
    # model = models.vgg16(pretrained=True)
    model = models.vgg16(weights='IMAGENET1K_V1')
    # model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg16', pretrained=True)

    # for param in model.features.parameters():
    #     param.requires_grad = False

    num_classes = 3

    model.classifier[-1] = nn.Linear(in_features=4096, out_features=num_classes, bias=True)

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
    # torch.backends.cudnn.benchmark = True
    # start = torch.cuda.Event(enable_timing=True)     
    # end = torch.cuda.Event(enable_timing=True)

    train_losses, test_losses, epoch_arr = [],[],[]
    train_accuracy_arr = []
    print_every = 100
    train_correct = 0 
    train_total = 0 
    steps = 0

    # start.record()
        
        # setting fine tuned parameters
        # params_to_update_1 = []
        # params_to_update_2 = []
        # params_to_update_3 = []

        # Not only output layer, "features" layers and other classifier layers are tuned.
        # update_param_names_1 = ["features"]
        # update_param_names_2 = ["classifier.0.weight",
        #                         "classifier.0.bias", "classifier.3.weight", "classifier.3.bias"]
        # update_param_names_3 = ["classifier.6.weight", "classifier.6.bias"]

        # # store parameters in list
        # for name, param in model.named_parameters():
        #     if update_param_names_1[0] in name:
        #         param.requires_grad = True
        #         params_to_update_1.append(param)
        #         #print("params_to_update_1:", name)

        #     elif name in update_param_names_2:
        #         param.requires_grad = True
        #         params_to_update_2.append(param)
        #         #print("params_to_update_2:", name)

        #     elif name in update_param_names_3:
        #         param.requires_grad = True
        #         params_to_update_3.append(param)
        #         #print("params_to_update_3:", name)

        #     else:
        #         param.requires_grad = False
                #print("no learning", name)
    learning_rate = 1e-4
    optimizer = optim.Adam(model.parameters(), 
                            lr=learning_rate,
                            )

    # Learning Rates
    # optimizer = optim.Adam([
    #     {'params': params_to_update_1, 'lr': 1e-4},
    #     {'params': params_to_update_2, 'lr': 5e-4},
    #     {'params': params_to_update_3, 'lr': 1e-3}
    #     ]
    # )
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
                           epochs=3)
#     # training loop
#     model.train()
#     for epoch in range(epochs):  # loop over the dataset multiple times
#         running_loss = 0.0
# #         print(f'training iter {epoch}')
#         for i, (inputs, labels) in enumerate(train_dataloader, 0):
#             # get the inputs; data is a list of [inputs, labels]
            
#             # move the inputs and labels tensor to cuda
#             inputs, labels = inputs.cuda(), labels.cuda()

#             # zero the parameter gradients
#             optimizer.zero_grad()
            
#             # forward pass
#             outputs = model(inputs)
#             # outputs = model.forward(inputs)
#             loss = criterion(outputs, labels)

#             # backward propagate
#             # loss.backward()
#             scheduler.step(loss)

#             # Accumulate the losses
#             # running_loss += loss.item()
#             running_loss += loss.item()

#             # Compute predictions and accuracy
#             # _, predicted = torch.max(outputs.data, 1)
#             _, predicted = torch.max(outputs.data, 1)
#             train_total += labels.size(0)
#             train_correct += (predicted == labels).sum().item()
#             # train_correct += torch.sum(predicted == labels.data).sum().cpu().numpy()
#             # if steps % print_every == 0 :
#             #     model.eval()
#             #     train_losses.append(running_loss/len(train_dataloader))
#             #     print(f"Epoch {epoch+1}/{epochs}.. "
#             #        f"Train loss: {running_loss/print_every:.3f}.. ")
#             #     running_loss = 0.0
#             #     model.train()
                
#             # if i % 100 == 99:    # print every 2000 mini-batches
#             #     print('[%d, %5d] loss: %.3f' %
#             #         (epoch + 1, i + 1, running_loss / 100))
#             #     print(f"Epoch {epoch+1}/{epochs}, Training Loss: {running_loss:.5f}")
#             #     running_loss = 0.0
        
#         running_loss /= (len(train_dataloader.dataset) * 100)
#         train_losses.append(running_loss)
#         print(f"Epoch {epoch+1}/{epochs}, Training Loss: {running_loss:.4f}")
        
#         train_accuracy = 100 * train_correct / train_total
#         train_accuracy = train_correct/len(train_dataloader.dataset)
#         print(f"training accuracy: {train_accuracy:.4f}")
#         train_accuracy_arr.append(train_accuracy)

#         epoch_arr.append(epoch)


    # train_accuracy = 100 * train_correct / train_total
    # train_accuracy_arr.append(train_accuracy)

    # Plot training loss
    # plt.style.use('ggplot')
    # plt.figure()
    # plt.plot(np.arange(0, len(train_losses)), train_losses, label='train_loss')

    # # plt.plot(epoch_arr, train_losses, label = 'Train Loss')
    # # plt.plot(epoch_arr, train_accuracy_arr, label='Train Accuracy')
    # plt.title('Training Loss on COVID-19 Dataset')
    # plt.xlabel('Epoch #')
    # plt.ylabel('Loss')
    # plt.legend(loc='lower left')
    # plt.savefig('training_loss.png')

    # plt.style.use('ggplot')
    # plt.figure()
    # plt.plot(np.arange(0, len(train_accuracy_arr)), train_accuracy_arr, label='train_loss')
    # # plt.plot(epoch_arr, train_losses, label = 'Train Loss')
    # # plt.plot(epoch_arr, train_accuracy_arr, label='Train Accuracy')
    # plt.title('Training Accuracy on COVID-19 Dataset')
    # plt.xlabel('Epoch #')
    # plt.ylabel('Accuracy')
    # plt.legend(loc='lower left')
    # plt.savefig('training_accuracy.png')
    
    # end.record()

    # # Waits for everything to finish running
    # torch.cuda.synchronize()

    # print('Finished Training')
    # print(start.elapsed_time(end))

    # correct = 0
    # total = 0
    # predicted_labels = []
    # true_labels = []
    # # test_loss = 0 
    # # accuracy = 0

    # # testing loop
    # model.eval()
    # with torch.no_grad():
    #     for (inputs, labels) in test_dataloader:
            
    #         images, labels = inputs.cuda(), labels.cuda()
    #         outputs = model(images)
    #         # outputs = model.forward(images)
            
    #         # batch_loss = criterion(outputs, labels)
    #         # ps = torch.exp(outputs)
    #         _, predicted = torch.max(outputs.data, 1)
    #         total += labels.size(0)
    #         # correct += (predicted == labels).sum().item()
    #         correct += torch.sum(predicted == labels.data)
    #         predicted_labels.extend(predicted.cpu().numpy())
    #         true_labels.extend(labels.cpu().numpy())
    #         # top_p, top_class = ps.topk(1, dim=1)
    #         # equals = top_class == labels.view(*top_class.shape)
    #         # test_loss += batch_loss.item()
    #         # accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
    #         # test_losses.append(test_loss/len(test_dataloader))
    #         # print(f"Test loss: {test_loss/len(test_dataloader):.3f}.."
    #         #       f"Test accuracy: {accuracy/len(test_dataloader):.3f}")
    # # Accuracy = metrics.accuracy_score(total, correct)

    # # print(f'what is accuracy: {Accuracy}')

    # print('Accuracy of the network on the 523 test images: %d %%' % (
    #     100 * correct / total))

    # target_names = ['0', '1', '2'] # Add your own class names

    # # Compute precision, recall, and f1-score
    # # precision = metrics.precision_score(true_labels, predicted_labels, average='weighted')
    # # recall = metrics.recall_score(true_labels, predicted_labels, average='weighted')
    # # f1 = metrics.f1_score(true_labels, predicted_labels, average='weighted')

    # # print(f'Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}')
    # report = metrics.classification_report(true_labels, predicted_labels, target_names=target_names)

    # print(report)
