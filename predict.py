import torch
import matplotlib.pyplot as plt
import numpy as np
import argparse
import json
from torch import nn, optim
from torchvision import datasets, transforms, models
from PIL import Image
from collections import OrderedDict

parser = argparse.ArgumentParser()
parser.add_argument('input', action="store", type=str)
parser.add_argument('checkpoint', action="store", type=str)
parser.add_argument('--category_names', action="store",
                    dest="category_names", type=str)
parser.add_argument('--top_k', action="store",
                    dest="top_k", type=int)
parser.add_argument('--gpu', action="store_true",dest='gpu',
                     default=False)

results = parser.parse_args()
input_im = results.input
gpu = results.gpu
device = torch.device("cuda" if (torch.cuda.is_available() and gpu) else "cpu")
top_k = results.top_k if results.top_k else 1

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    arch = checkpoint['model']
    if arch=='vgg16':
        model = models.vgg16(pretrained=True)
    elif arch=='vgg13':
        model = models.vgg13(pretrained=True)
    else:
        model = models.vgg19(pretrained=True)
    model.classifier = checkpoint['classifier']
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['model_state_dict'])
    return model
model = load_checkpoint(results.checkpoint)
model.to(device)
mean = np.array([0.485, 0.456, 0.406])
std=np.array([0.229, 0.224, 0.225])
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    transform = transforms.Compose([transforms.Resize(256),transforms.CenterCrop(224),transforms.ToTensor()
                                   ,transforms.Normalize(mean=mean,std=std)])
    image = transform(image)
    return image.numpy()
def predict(image_path, model, topk=1):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    with torch.no_grad():
        pil_image = Image.open(image_path)
        image = process_image(pil_image)
        image = torch.from_numpy(image).view([1,3,224,224])
        image = image.to(device)
        logps = model(image)
        ps = torch.exp(logps)
        top_p, top_guess = ps.topk(topk, dim=1)
        class_to_idx = model.class_to_idx
        convert = { class_to_idx[k]:k for k in class_to_idx}
        converted_guesses = []
        top_guess,top_p = top_guess.cpu(),top_p.cpu()
        for x in top_guess.numpy().reshape(topk,1):
            converted_guesses.append(convert[x[0]])
    return top_p.numpy().reshape(topk), converted_guesses

probs, classes = predict(input_im,model,top_k)
names = []
if results.category_names:
    with open(results.category_names, 'r') as f:
        cat_to_name = json.load(f)
    for x in classes:
        names.append(cat_to_name[x])
else:
    names = classes
for i in range(top_k):
    print(f'{i+1}) {names[i]}, {int(probs[i]*10000)/100}%')
    
