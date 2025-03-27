import torch
from torchvision import models, transforms
from torch import nn
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
#import train as test_dir
import json

test_dir = 'flowers/test' 

def LoadModel(path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = models.vgg16(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    classifier = nn.Sequential(nn.Linear(25088, 1024),
                               nn.ReLU(),
                               nn.Dropout(0.2),
                               nn.Linear(1024, 500),
                               nn.ReLU(),
                               nn.Linear(500, 102),
                               nn.LogSoftmax(dim=1))
    model.classifier = classifier
    #map_location = 'cuda' if torch.cuda.is_available() else 'cpu'
    checkpoint = torch.load(path, map_location=device)
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    return model

model = LoadModel('checkpoint.pth')

def process_image(imagePath):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns a tensor
    '''
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = Image.open(imagePath)
    input_tensor = preprocess(image)
    return input_tensor

def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax

image = process_image(test_dir + '/1/image_06743.jpg')
imshow(image)

def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image = process_image(image_path)
    image = image.unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        output = model(image)
    probs, classes = torch.topk(torch.exp(output), topk)
    probs = probs.cpu().numpy()[0]
    classes = classes.cpu().numpy()[0]
    return probs, classes

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
   
    image_path = test_dir + '/1/image_06743.jpg'
    # Load model
    model = LoadModel('checkpoint.pth')
    probs, classes = predict(image_path, model)
    print(image_path)
    print(probs)
    print([cat_to_name[str(index)] for index in classes])