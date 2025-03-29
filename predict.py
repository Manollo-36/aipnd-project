import torch
from torchvision import models, transforms
from torch import nn
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import json
import argparse


def get_input_args():    
    # command lines for user input 
    parser = argparse.ArgumentParser(description='train.py')
    parser.add_argument('--checkpoint', type=str, default='checkpoint.pth', help='path to save checkpoint')
    # data directory
    parser.add_argument("--image_path", type=str, default= './flowers/test/1/image_06743.jpg', help='Path to flower image')
    parser.add_argument('--model_arch', type=str, default='VGG19', help='model architecture')
    parser.add_argument('--category_names', type=str, default='./cat_to_names.json', help='model architecture')
    parser.add_argument('--top_k', type=int, default=5, help='learning rate')
    parser.add_argument('--gpu', action='store_true', default=False, help='use GPU for training')
    parser.add_argument('--hidden_units', type=int,  nargs=2, default=[1024,500], help='number of hidden units')
    return parser.parse_args()

args = get_input_args()
#checkpoint_path = 'checkpoint.pth'

def LoadModel(path):
    device = torch.device('cuda' if torch.cuda.is_available() and args.gpu else 'cpu')
    if args.model_arch == 'VGG19':
        model = models.vgg19(weights="VGG19_Weights.DEFAULT")
    else:
         model = models.vgg16(weights="VGG164_Weights.DEFAULT")
    for param in model.parameters():
        param.requires_grad = False
    classifier = nn.Sequential(nn.Linear(25088, args.hidden_units[1]),
                               nn.ReLU(),
                               nn.Dropout(0.2),
                               nn.Linear(args.hidden_units[0], out_features=args.hidden_units[1]),
                               nn.ReLU(),
                               nn.Linear(args.hidden_units[0], [1]),
                               nn.LogSoftmax(dim=1))
    model.classifier = classifier
    checkpoint = torch.load(f=path,map_location=device,weights_only=False)
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])  
    model.to(device)
    return model

model = LoadModel(args.checkpoint)

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
    image = Image.open(args.image_path)
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

image = process_image(args.image_path)
imshow(image)

def predict(image_path, model,top_k):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    device = torch.device("cuda" if torch.cuda.is_available()and args.gpu else "cpu")
    image = process_image(args.image_path)
    image = image.unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        output = model(image)
    probs, classes = torch.topk(torch.exp(output), top_k)
    probs = probs.cpu().numpy()[0]
    classes = classes.cpu().numpy()[0]
    return probs, classes

with open(args.category_names, 'r') as f:
    cat_to_name = json.load(f)   
    image_path = args.image_path
    # Load model
    model = LoadModel(args.checkpoint)
    probs, classes = predict(args.image_path, model,args.top_k)
    print(image_path)
    print(probs)
    #print([cat_to_name[str(index)] for index in classes])