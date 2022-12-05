import torch
import matplotlib.pyplot as plt
import numpy as np
import argparse
from torch import nn, optim
from torchvision import datasets, transforms, models
from PIL import Image
from collections import OrderedDict
from workspace_utils import keep_awake
parser = argparse.ArgumentParser()
parser.add_argument('data_dir', action="store", type=str)
parser.add_argument('--save_dir', action="store",dest = 'save_dir',
                    type=str)
parser.add_argument('--arch', action="store",
                 dest='arch',type=str, default = "test", help='vgg13, vgg16 or vgg19 by default')
parser.add_argument('--learning_rate', action="store",
                    dest="lr", type=float)
parser.add_argument('--hidden_units', action="store",
                    dest="hidden_units", type=int)
parser.add_argument('--epochs', action="store", dest="epochs", type=int)
parser.add_argument('--gpu', action="store_true",dest='gpu',
                     default=False)

results = parser.parse_args()
save_dir = results.save_dir 
arch = results.arch if results.arch else 'vgg19'
lr = results.lr if results.lr else 0.001
hidden_units = results.hidden_units if results.hidden_units else 4
epochs = results.epochs if results.epochs else 10
gpu = results.gpu

data_dir = results.data_dir
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

mean = [0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]

data_transforms = transforms.Compose([transforms.RandomVerticalFlip(0.1),
                                      transforms.RandomHorizontalFlip(0.05),
            transforms.RandomRotation(60),transforms.Resize(255),transforms.CenterCrop(224),transforms.ToTensor(),
                                     transforms.Normalize(mean=mean,std=std)
                                     ])

real_transforms = transforms.Compose([
    transforms.Resize(255),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean,std=std)
])


train_datasets = datasets.ImageFolder(train_dir,transform=data_transforms)
test_datasets = datasets.ImageFolder(test_dir,transform=real_transforms)
valid_datasets = datasets.ImageFolder(valid_dir,transform=real_transforms)

trainloader = torch.utils.data.DataLoader(train_datasets,batch_size=64,shuffle=True )
testloader = torch.utils.data.DataLoader(test_datasets,batch_size=64,shuffle=True )
validloader = torch.utils.data.DataLoader(valid_datasets,batch_size=64,shuffle=True )

if arch=='vgg16':
    model = models.vgg16(pretrained=True)
elif arch=='vgg13':
    model = models.vgg13(pretrained=True)
else:
    model = models.vgg19(pretrained=True)

device = torch.device("cuda" if (torch.cuda.is_available() and gpu) else "cpu")

for param in model.parameters():
    param.requires_grad = False

    
    
total_data = 25088
ordered_dict = OrderedDict()
if hidden_units>2:
    ordered_dict['fc'] = nn.Linear(total_data,2048)
    ordered_dict['relu'] = nn.ReLU()
    ordered_dict['dropout'] = nn.Dropout(0.5)
    ordered_dict[f'fc1'] = nn.Linear(2048, 1024)
    ordered_dict[f'relu2'] = nn.ReLU()
    ordered_dict[f'dropout3'] = nn.Dropout(0.5)
    for i in range(hidden_units-2):
        ordered_dict[f'fc{i+2}'] = nn.Linear(1024, 1024)
        ordered_dict[f'relu{i+2}'] = nn.ReLU()
        ordered_dict[f'dropout{i+2}'] = nn.Dropout(0.5)
else:
    ordered_dict['fc'] = nn.Linear(total_data,1024)
    ordered_dict['relu'] = nn.ReLU()
    ordered_dict['dropout'] = nn.Dropout(0.5)
output_size = 102
ordered_dict['output']= nn.Linear(1024, output_size)
ordered_dict['softmax']= nn.LogSoftmax(dim=1)
classifier = nn.Sequential(ordered_dict)


model.classifier = classifier

model.to(device)



criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=lr)
print_every = 10
running_loss = 0
step=0

for e in keep_awake(range(epochs)):
    step = 0
    for images, labels in trainloader:
        step+=1
        images, labels = images.to(device), labels.to(device)

        logps = model.forward(images)
        loss = criterion(logps, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if step % print_every == 0:
            model.eval()
            test_loss = 0
            accuracy = 0
            with torch.no_grad():
                for images, labels in validloader:
                    images, labels = images.to(device), labels.to(device)
                    logps = model.forward(images)
                    ps = torch.exp(logps)
                    test_loss += criterion(logps,labels).item()
                    top_p, top_guess = ps.topk(1, dim=1)
                    equals = top_guess == labels.view(*top_guess.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                print(f'Epoch {e+1}/{epochs}, Step:{step}')
                print(f'Training Loss: {running_loss/print_every}')
                print(f'Testing Loss: {test_loss/len(validloader)}')
                print(f'Accuracy: {accuracy/len(validloader)}')
                running_loss = 0
                model.train()
if save_dir:
    checkpoint = { 'classifier': classifier,
                  'optimizer_state_dict': optimizer.state_dict(),
                  'model_state_dict': model.state_dict(),
                  'epochs': epochs,
                  'model': arch,
                  'lr': lr,
                  'class_to_idx': train_datasets.class_to_idx
             }
    torch.save(checkpoint, f'{save_dir}/checkpoint.tar')
    print(f'Saved to {save_dir}/checkpoint.tar')

                    
        


    