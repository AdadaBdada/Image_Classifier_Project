import torch
import json
import torch.nn as nn
from collections import OrderedDict
import numpy as np
import argparse
from torchvision import datasets, models, transforms, utils
from PIL import Image

use_gpu = torch.cuda.is_available

def process_image(image_path):
    '''
    Scales, crops, and normalizes a PIL image for a PyTorch
    model, returns an Numpy array
    '''
    # Open the image
    img = Image.open(image_path)

    # Resize
    if img.size[0] > img.size[1]:
        img.thumbnail((10000, 256))
    else:
        img.thumbnail((256, 10000))

    # Crop
    left_margin = (img.width - 224) / 2
    bottom_margin = (img.height - 224) / 2
    right_margin = left_margin + 224
    top_margin = bottom_margin + 224
    img = img.crop((left_margin, bottom_margin, right_margin,
                    top_margin))

    # Normalize
    img = np.array(img) / 255  # model expected floats 0–1, scaling by 255.
    mean = np.array([0.485, 0.456, 0.406])  # mean
    std = np.array([0.229, 0.224, 0.225])  # std
    img = (img - mean) / std

    # Move color channels to first dimension as expected by PyTorch
    img = img.transpose((2, 0, 1))

    return img

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = models.vgg16()

    model.class_to_idx = checkpoint['class_to_idx']
    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(25088, 4096)),
        ('relu', nn.ReLU()),
        ('fc2', nn.Linear(4096, 102)),
        ('drop1', nn.Dropout(p=0.50)),
        ('output', nn.LogSoftmax(dim=1))
    ]))

    model.classifier = classifier
    model.load_state_dict(checkpoint['state_dict'])

    model = load_checkpoint('classifier.pth')
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    return model, class_to_idx, idx_to_class


def predict(image_path, model, top_num=5):
    # Process image
    img = process_image(image_path)

    # Numpy -> Tensor (expect troch.DoubleTesor but found type torch.cuda.FloatTensor so type(torch.FloatTensor))
    image_tensor = torch.from_numpy(img).type(torch.FloatTensor)
    # supposed to be the batch size, but right now only one picture, so add a 1 as the first argument of our tensor
    model_input = image_tensor.unsqueeze(0)

    # Probs, previous use logsoftmax, here use exponential to cenvert it as original probability
    probs = torch.exp(model.forward(model_input))

    # Top probs
    top_probs, top_labs = probs.topk(top_num)
    top_probs = top_probs.detach().numpy().tolist()[0]
    top_labs = top_labs.detach().numpy().tolist()[0]

    # Convert indices to classes
    idx_to_class = {val: key for key, val in model.class_to_idx.items()}
    # pull the top classes we need to index
    top_labels = [idx_to_class[lab] for lab in top_labs]
    top_flowers = [cat_to_name[idx_to_class[lab]] for lab in top_labs]
    return top_probs, top_labels, top_flowers

if __name__ == "__main__":


    parser = argparse.ArgumentParser()

    parser.add_argument('--input_img', default='./flowers/test/10/image_07090.jpg', nargs='*', action="store",type=str)
    parser.add_argument('--checkpoint', default='./classifier.pth',nargs='*', action="store",type=str)
    parser.add_argument('--top_k', default=5, dest="top_k",action='store', type=int)
    parser.add_argument('--category_names',dest='category_names',action='store', default='cat_to_name.json')
    parser.add_argument('--gpu',default='gpu', action='store', dest= "gpu")

    args = parser.parse_args()

    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)
    
    print('------start loading------')
    model, class_to_idx, idx_to_class = load_checkpoint(args.checkpoint)
    print('------loading finished------')
    print(model)
    print(class_to_idx)
    
    
    top_probs, top_labels, top_flowers = predict(args.input_image,args.top_k)
    print('Predicted Classes: ', top_labels)
    print('Class Names: ')
    [print(cat_to_name[x]) for x in top_labels]
    print('Predicted Probability: ', top_probs)
