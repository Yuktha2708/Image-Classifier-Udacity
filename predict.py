print('Predict.py is running!')

import argparse
import matplotlib.pyplot as plt
import seaborn as sb
import torch
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import json
from PIL import Image
from args_predict import args

hyper_params = { 'image_dir': args.image_path,'checkpoint_dir': args.checkpoint_path,'category_dir': args.cat_path, 'topk': args.topk,'gpu': args.gpu}


cat_name = hyper_params['category_dir']
with open(cat_name, 'r') as f:
    cat_to_name = json.load(f)
print('Category names:' + str(len(cat_to_name)))



def load_model(checkpoint):
    arch = checkpoint['arch']
    out_layers = checkpoint['out_layers']
    print('\nout_layers = ' + str(out_layers))
    hidden_layers = checkpoint['hidden_layers']
    in_layers = checkpoint['in_layers']
    dropout = checkpoint['dropout']
    classifier_type = checkpoint.get('classifier_type', 'double')
    
    if checkpoint['arch'] == 'vgg16':
        model = models.vgg16(pretrained=True)
    elif (checkpoint['arch'] == 'densenet121'):
        model = models.densenet121(pretrained = True)
    elif (checkpoint['arch'] == 'alexnet'):
        model = models.alexnet(pretrained = True)
        
        model.eval()
        for param in model.parameters():
            param.requires_grad = False

    model.class_to_idx = checkpoint['class_to_idx']
    

    from collections import OrderedDict
    if classifier_type == 'single':
        classifier = nn.Sequential(OrderedDict([
            ('dropout', nn.Dropout(dropout)),
            ('fc', nn.Linear(in_layers, out_layers)),
            ('out', nn.LogSoftmax(dim=1))
        ]))
    elif classifier_type == 'double':
        classifier = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(in_layers, hidden_layers)),
            ('relu', nn.ReLU()),
            ('dropout', nn.Dropout(dropout)),
            ('fc2', nn.Linear(hidden_layers, out_layers)),
            ('output', nn.LogSoftmax(dim=1))
        ]))
    else:
        raise ValueError('Invalid classifier type: {}'.format(classifier_type))

    model.classifier = classifier
    model.load_state_dict(checkpoint['state_dict'])
    
    return model


checkpoint = torch.load(hyper_params['checkpoint_dir'])
model = load_model(checkpoint)



def process_images(image_path):

    image_transformer = transforms.Compose([transforms.Resize(256),transforms.CenterCrop(224),transforms.ToTensor()])                                  
    prep_image = Image.open(image_path)
    prep_image = image_transformer(prep_image)
    np_image = np.array(prep_image)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np.transpose(np_image, (1, 2, 0)) - mean)/std    
    np_image = np.transpose(np_image, (2, 0, 1))
    return np_image


image_path = hyper_params['image_dir']
image = process_images(image_path)


def predict(image_path, model, topk_number, cat_to_name, gpu):
    
    model.eval()
    image = process_images(image_path)
    tensor_image = torch.from_numpy(image).type(torch.FloatTensor)
    
    gpu_mode = False
    if (gpu == 'GPU' and torch.cuda.is_available()):
        gpu_mode = True 
        print('training in cuda')
        model.to('cuda')
        tensor_image = tensor_image.cuda()    
       
    elif (gpu == 'CPU'):
        print('training in CPU')
        model.to('cpu')
        

    tensor_image = tensor_image.unsqueeze_(0)
        

    probability = torch.exp(model.forward(tensor_image))
    

    prob, top_classes = probability.topk(topk_number)
    
    if gpu_mode:
        prob = prob.cpu().detach().numpy().tolist()[0]
        top_classes = top_classes.cpu().detach().numpy().tolist()[0]
    else:
        prob = prob.detach().numpy().tolist()[0]
        top_classes = top_classes.detach().numpy().tolist()[0]
    

    idx_to_class = {val: key for key, val in model.class_to_idx.items()}
    
    labels = [idx_to_class[classes] for classes in top_classes]
    top_fl = [cat_to_name[idx_to_class[classes]] for classes in top_classes] 
    
    return prob, top_fl


checkpoint = torch.load(hyper_params['checkpoint_dir'])
model = load_model(checkpoint)


prob, top_fl = predict(image_path, model, hyper_params['topk'], cat_to_name, hyper_params['gpu'] )
print("\nHere's the prediction:")
print(top_fl)
print('\nwith probabilities:')
print(prob)