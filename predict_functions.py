#imports

import argparse
from collections import OrderedDict
from itertools import chain
import json
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import pandas as pd
import torch
import torch.nn.functional
import torchvision.models as models
from torch import nn
from torch import optim


def get_input_arguments():
    """This function takes user input and returns
    a data structure which stores the command line 
    arguments.
    
    Parameters: 
    --
    Return values:
    -image path (mandatory)
    -checkpoint (mandatory)
    -top_k (k most likely predictions) 
    -category_names: if True, creates a maping to real names
    -device: switch between cpu and gpu/cuda
    """
    # Creates parse 
    parser = argparse.ArgumentParser(description="Process user input.")
    
    #add arguments
    parser.add_argument("image_path", help = "Path to image")#  use e.g. "ImageClassifier/flowers/test/1/image_06743.jpg"
    parser.add_argument("checkpoint", help = "Checkpoint of pre-trained model") #use e.g. "checkpoint.pth"
    parser.add_argument("--top_k", help = "if True, creates a maping to real names", default = 3)
    parser.add_argument("--GPU", help = "turn cuda on", default = False)
    
    #use default argument for image path to make testing more convenient
    #REMOVE AFTER TESTING!!!
    #parser.add_argument("--image_path", help = "Path to image", default = "ImageClassifier/flowers/test/1/image_06743.jpg")
    
    
    #returns parsed argument collection
    return parser.parse_args()


def load_checkpoint(path, device):
    """This function loads a model from a checkpoint & returns it
    Parameters:
    -path: path of the checkpoint
    Returns:
    -loaded model"""
    #load model with cpu/cuda
    if device == "cpu":
        checkpoint = torch.load(path, map_location=lambda storage, loc: storage)
    elif device == "cuda":
        checkpoint = torch.load(path)
    
    model = checkpoint["model"]
    model.classifier = checkpoint["classifier"]
    model.class_to_idx = checkpoint["class_to_idx"]
    model.load_state_dict(checkpoint["state_dict"])
    return model


def process_image(image):
    """This function scales, crops, and normalizes a PIL image for a PyTorch model & returns an Numpy array
    Parameters:
    -image path
    Returns:
    -np_image: preprocessed numpy array of the image"""        
    #open image
    image = Image.open(image)
    #identify image dimensions, set target length
    width, height = image.size
    target_length = 256    
    #scale image dimensions
    if width > height:
        width = int(width * target_length / height)
        height = target_length       
    elif width < height:
        height = int(height * target_length / width)
        width = target_length     
    #resize image
    image = image.resize((width, height))      
    #crop image
    crop_length = 224
    start_x = (width - crop_length) / 2
    end_x = start_x + crop_length
    start_y = (height - crop_length) / 2
    end_y = start_y + crop_length
    image = image.crop((start_x, start_y, end_x, end_y))        
    #scale color channels
    np_image = np.array(image)/255    
    #transpose image
    means = np.array([0.485, 0.456, 0.406])
    stds = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - means) / stds 
    np_image = np_image.transpose((2, 0, 1))
    
    return np_image

def predict_image(np_image, model, top_k = 3):
    """This function predicts one or several classes for a certain image
    Parameters:
    -np_image: preprocessed numpy array of the image
    -model: pretrained model
    -top_k: number of classes which should be predicted (default == 1 class)
    Returns:
    -classes: list of predicted classes
    -probabilities: list of predicted probabilities"""
    #change data types
    top_k = int(top_k)
    image_tensor = torch.FloatTensor(np_image)
    image_tensor.unsqueeze_(0)
    #predict values for classes
    with torch.no_grad():
        output = model.forward(image_tensor)    
    #convert output values to propabilities
    probability_prediction = torch.nn.functional.softmax(output, dim = 1)
    #select top k probabilities & their classes
    probabilities, class_numbers = torch.topk(probability_prediction, top_k)
    #convert class numbers to class labels
    class_to_idx = model.class_to_idx
    idx_to_class = {str(value) : int(key) for key, value in class_to_idx.items()}
    #print(idx_to_class)
    #convert probability tensor to numpy array
    probabilities = probabilities[0][:].tolist()
    #convert top class numbers to classes labels
    classes = np.array([idx_to_class[str(index)] for index in class_numbers.numpy()[0]])
    classes = list(classes)
    classes = [str(i) for i in classes]

    return classes, probabilities


def get_label_mapping(filename):
    """This function loads the label mapping file"""
    with open (filename, "r") as file:
        loaded_file = json.load(file)
    return loaded_file

def get_class_names(classes, probabilities, cat_to_name):
    """This function creates a dictionary of the predicted classes & their probabilities
    Parameters:
    -classes: class IDs
    -probabilities
    Returns:
    -prediction_dict: a dictionary with real class names & their probabilities"""
    labels_y = []
    prediction_dict = {}
    for key in classes:
        labels_y.append(cat_to_name[key])
    
    for flower, flower_probabilitiy in zip(labels_y, probabilities):
        prediction_dict[flower] = round(flower_probabilitiy, 2)
            
    return prediction_dict