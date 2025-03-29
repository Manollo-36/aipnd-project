import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models
import argparse
import json

# data_dir = 'flowers'
# train_dir = data_dir + '/train'
# valid_dir = data_dir + '/valid'
# test_dir = data_dir + '/test'

def get_input_args():    
    # command lines for user input 
    parser = argparse.ArgumentParser(description='train.py')
    parser.add_argument('--save_dir', type=str, default='checkpoint.pth',required=True, help='path to save checkpoint')
    # data directory
    parser.add_argument("--data_dir", type=str, default= './flowers', help='Path to folder of flower images')
    parser.add_argument("--train_dir", type=str, default= './flowers/train', help='Path to folder of train images')
    parser.add_argument("--valid_dir", type=str, default= './flowers/valid', help='Path to folder of valid images')
    parser.add_argument("--test_dir", type=str, default= './flowers/test', help='Path to folder of test images')
    parser.add_argument('--model_arch', type=str, default='VGG19', help='model architecture')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
    parser.add_argument('--hidden_units', type=int,nargs=2, default=[1024,500], help='number of hidden units')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs')
    parser.add_argument('--gpu', action='store_true', default=False, help='use GPU for training')
    return parser.parse_args()
args = get_input_args()

train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])
valid_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])
test_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

data_transforms = [train_transforms, valid_transforms, test_transforms]
train_data = datasets.ImageFolder(args.train_dir, transform=train_transforms)
valid_data = datasets.ImageFolder(args.valid_dir, transform=valid_transforms)
test_data = datasets.ImageFolder(args.test_dir, transform=test_transforms)
image_datasets = [train_data, valid_data, test_data]
trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
validloader = torch.utils.data.DataLoader(valid_data, batch_size=64)
testloader = torch.utils.data.DataLoader(test_data, batch_size=64)
dataloaders = [trainloader, validloader, testloader]

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
    device = torch.device("cuda" if torch.cuda.is_available() and args.gpu else "cpu")
    if args.model_arch == 'VGG19':
        model = models.vgg19(weights="VGG19_Weights.DEFAULT")
    else:
        model = models.vgg16(weights="VGG16_Weights.DEFAULT")
    for param in model.parameters():
        param.requires_grad = False
        
        model.classifier = classifier = nn.Sequential(nn.Linear(25088, args.hidden_units[1]),
                                nn.ReLU(),
                                nn.Dropout(0.2),
                                nn.Linear(args.hidden_units[0],args.hidden_units[1]),
                                nn.ReLU(),
                                nn.Linear(args.hidden_units[0], args.hidden_units[1]),
                                nn.LogSoftmax(dim=1))
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), args.learning_rate)
    model.to(device)
    epochs = args.epochs
    steps = 0
    running_loss = 0
    print_every = 5
    for epoch in range(epochs):
        for inputs, labels in trainloader:
            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
            if steps % print_every == 0:
                test_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in testloader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)
                        
                        test_loss += batch_loss.item()
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                        
                print(f"Epoch {epoch+1}/{epochs}.. "
                    f"Train loss: {running_loss/print_every:.3f}.. "
                    f"Test loss: {test_loss/len(testloader):.3f}.. "
                    f"Test accuracy: {accuracy/len(testloader):.3f}")
                running_loss = 0
                model.train()
    model.class_to_idx = train_data.class_to_idx
    torch.save( {'hidden layer': 1024,
                'output_size': 102,
                'dropout': 0.5,
                'epochs': args.epochs,#10,
                'model architecture': args.model_arch,#'vgg19',
                'optimizer': optimizer.state_dict(),
                'classifier': model.classifier,
                'learning_rate': args.learning_rate ,#0.001,
                'state_dict': model.state_dict(),
                'class_to_idx': model.class_to_idx}, 'checkpoint.pth')   
