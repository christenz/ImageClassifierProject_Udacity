#imports

import argparse
from collections import OrderedDict
import json
import time
import torch
from torch import nn
from torch import optim
from torchvision import datasets
from torchvision import transforms
import torchvision.models as models



def get_input_arguments():
    """This function takes user input and returns
    a data structure which stores the command line 
    arguments.
    
    Parameters: 
    --
    Return values:
    -data_dir
    -arg
    -learning_rate
    -hidden_units_1
    -hidden_units_2
    -epochs
    -device
    """
    # Creates parse 
    parser = argparse.ArgumentParser(description="Process user input.")
    
    #add arguments
    parser.add_argument("data_dir", help = "Directory of the input data") #Use "ImageClassifier/flowers"
    parser.add_argument("--arg", help = "Set a model architecture", default = "VGG16")
    parser.add_argument("--learning_rate", help = "Set a learning rate", default = 0.001)
    parser.add_argument("--hidden_units_1", help = "Set the number of nodes in the 1st hidden unit", default = 4096)
    parser.add_argument("--hidden_units_2", help = "Set the number of nodes in the 2nd hidden unit", default = 1024)
    parser.add_argument("--epochs", help = "Set the number of epochs", default = 5)
    parser.add_argument("--GPU", help = "Turn cuda on", default=False,)
    
    #returns parsed argument collection
    return parser.parse_args()


def get_data_loader(data_dir):
    """This function loads the data for train/valid/test, transforms it & defines the dataloaders.
    Parameters: 
    -data_dir: directory of the training/validation/test data
    -batch_size_train: batch size for training dataloader
    -batch_size_valid_test: batch size for validation and test dataloader
    
    Return values:
    -train_loader
    -valid_loader
    -test_loader
    -train_datasets
    """
    #set batch sizes
    batch_size_train=128
    batch_size_valid_test=64
    #define directories
    train_dir = data_dir + "/train"
    valid_dir = data_dir + "/valid"
    test_dir = data_dir + "/test"    
    #define transformations
    train_transforms = transforms.Compose([transforms.RandomResizedCrop(224),
                                           transforms.RandomRotation(20),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])
    
    valid_test_transforms = transforms.Compose([transforms.Resize(256),
                                                transforms.CenterCrop(224),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])
    #load datasets
    train_datasets = datasets.ImageFolder(train_dir, transform = train_transforms)
    valid_datasets = datasets.ImageFolder(valid_dir, transform = valid_test_transforms)
    test_datasets = datasets.ImageFolder(test_dir, transform = valid_test_transforms)
    
    #define dataloaders
    train_loader = torch.utils.data.DataLoader(train_datasets, batch_size_train, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_datasets, batch_size_valid_test)
    test_loader = torch.utils.data.DataLoader(test_datasets, batch_size_valid_test)
    
    return train_loader, valid_loader, test_loader, train_datasets




 
def get_model(model, hidden_units_1=2048, hidden_units_2=1024):
    """This function creates the model structure
    Parameters:
    -model: torchvision model (needs to be included in the dictionary!)
    -hidden_units_1: no. of hidden units in the first layer
    -hidden_units_2: no. of hidden units in the first layer
    Returns:
    -model
    """
    #get data types right
    hidden_units_1 = int(hidden_units_1)
    hidden_units_2 = int(hidden_units_2)
    #define models allowed here
    alexnet = models.alexnet(pretrained=True)
    densenet = models.densenet161()
    resnet18 = models.resnet18(pretrained=True)
    squeezenet = models.squeezenet1_0(pretrained=True)
    VGG16 = models.vgg16(pretrained=True)
    
    model_dic = {"alexnet": alexnet, "densenet": densenet, "squeezenet": squeezenet,"resnet": resnet18,  "VGG16": VGG16}
    current_model = model_dic[model]
    
    #turn off gradient for the features for speed up
    for parameter in current_model.parameters():
        parameter.requires_grad = False
    
    number_of_inputs = current_model.classifier[0].in_features
    
    classifier = nn.Sequential(OrderedDict([ 
    ('fc1', nn.Linear(number_of_inputs, hidden_units_1, bias=True)),
    ('relu', nn.ReLU()),
    ('dropout1', nn.Dropout()),
    
    ('fc2', nn.Linear(hidden_units_1, hidden_units_2, bias=True)),
    ('relu2', nn.ReLU()),
    ('dropout2',nn.Dropout()),
        
    ('fc3',nn.Linear(hidden_units_2, 102, bias=True)),
    ('output', nn.LogSoftmax(dim=1))
    ]))
    
    current_model.classifier = classifier
    
    return current_model


def get_model_evaluation(model, loader, criterion, device):
    """This model computes accuracy & loss of the model
    Parameters:
    -model
    -data loader
    -criterion: loss function
    -device: cpu or gpu
    
    Returns:
    -loss
    -accuracy"""
    model.eval()
    model.to(device)
    loss = 0
    accuracy = 0
    for images, labels in loader:
        #sent data to cuda if gpu mode is active
        images, labels = images.to(device), labels.to(device)     
        #feed forward, loss calculation
        output = model.forward(images)
        loss += criterion(output, labels).item()
        #compare output & true values
        output_exp = torch.exp(output)
        equality_check = (labels.data == output_exp.max(dim=1)[1])
        accuracy += equality_check.type_as(torch.FloatTensor()).mean()
    accuracy = accuracy / len(loader)
    loss = loss / len(loader)
            
    return  accuracy, loss


def train_model(model, train_loader, validation_loader, device = "cuda", criterion= nn.NLLLoss(), 
                optimizer = optim.Adam, learning_rate= 0.001, epochs = 1, print_every = 20):
    """This function trains the model
    Parameters:
    -model: structured model
    -device: cpu or gpu/cuda
    -criterion: loss function
    -optimizer: backpropagation function
    learning_rate
    -epochs
    -print_every: interval for training status report
    
    Returns:
    -trained model"""
    #ensure the data types are alright
    learning_rate = float(learning_rate)
    epochs = int(epochs)
    #switch between cpu/gpu
    if device == "cpu":
        current_device = torch.device = "cpu"
    elif device == "cuda":
        current_device = torch.device = "cuda:0"
    print(current_device)
    #set optimizer
    current_optimizer = optimizer(model.classifier.parameters(), learning_rate)
    steps= 0
    model.train()
    model.to(current_device)
    
    #start training
    time_0 = time.time()
    for e in range(epochs):
        running_loss = 0
        print(time.time() - time_0)
        for images, labels in train_loader:
            model.train()
            steps +=1
            #sent data to cuda if gpu mode is active
            images, labels = images.to(current_device), labels.to(current_device)
            #reset gradient to zero
            current_optimizer.zero_grad()   
            #feed forward
            outputs = model.forward(images)
            #loss function
            loss= criterion(outputs, labels)
            #backpropagation
            loss.backward()
            #weight update
            current_optimizer.step()
            #loss value
            running_loss += loss.item()
            
            #print status for every X steps
            if steps % print_every == 0:
                with torch.no_grad():
                    accuracy, loss = get_model_evaluation(model, validation_loader, criterion, current_device)
                print("Epoch: {}/{}... ".format(e+1, epochs),
                  "Training Loss: {:.2f}".format(running_loss/print_every),
                     "Validation Loss: {:.2f}.. ".format(loss),
                      "Validation Accuracy: {:.2f}".format(accuracy))
                #reset running loss after printing
                running_loss = 0
            model.train()
      #training loop
    print("training completed")
    time_0 = time.time()
    return model


def test_model(model, test_loader, device = "cuda", criterion= nn.NLLLoss()):
    """This function tests the model on the test dataset
    Parameter:
    -model
    -test loader
    -device
    -criterion: loss function
    
    Returns:
    test accuracy, test loss"""
    with torch.no_grad():
        test_accuracy, test_loss = get_model_evaluation(model, test_loader, criterion, device)
    return test_accuracy, test_loss


def save_dictionary(model, train_datasets, checkpoint_filename):
    """This functions saves the trained model to a dictionary
    Parameter:
    -model
    
    Returns:
    --
    """
    #current_device = torch.device(device)
    #model.to(device)
    
    checkpoint = {
    "model" : model,
    "state_dict" : model.state_dict(),
    "classifier" : model.classifier,
    "class_to_idx" : train_datasets.class_to_idx,
    "classifier.state_dict" : model.classifier.state_dict(),  
    }
    
    torch.save(checkpoint, checkpoint_filename)