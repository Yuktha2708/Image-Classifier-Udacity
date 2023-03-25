print('train.py is running!')

import argparse
import seaborn as sb
import torch
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from PIL import Image
from args_train import args
hyper_params = { 'data_dir': args.data_dir,
                    'save_dir': args.save_dir,
                    'arch': args.arch,
                    'epochs': args.epochs,
                    'lr': args.lr,
                    'gpu': args.gpu,
                    'in_layers': args.in_layers,
                    'hidden_layers': args.hidden_layers,
                    'out_layers': args.out_layers,
                    'drop': args.drop_rate,
                    'topk':args.topk
}


train_dir = hyper_params['data_dir'] + '/train'
valid_dir = hyper_params['data_dir'] + '/valid'
test_dir = hyper_params['data_dir'] + '/test'

normalize_dict = {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}

data_trans = {
    'train': transforms.Compose([
        transforms.RandomRotation(30),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(**normalize_dict)]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(**normalize_dict)]),
    'valid': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(**normalize_dict)])
}

dirs = {'train': train_dir,'test': test_dir,'valid': valid_dir}

images_datasets = {x: datasets.ImageFolder(dirs[x],transform = data_trans[x]) 
                  for x in list(data_trans.keys())}

data_load = {
'train_load' : torch.utils.data.DataLoader(images_datasets['train'], batch_size=64, shuffle=True),
'test_load' : torch.utils.data.DataLoader(images_datasets['test'], batch_size=64, shuffle=False),
'valid_load' : torch.utils.data.DataLoader(images_datasets['valid'], batch_size=64, shuffle=True)
}

images, labels = next(iter(data_load['test_load']))
print('test_load data: '+ str(len(images[0,2])))


def model_get(model_arch):
    
    if (model_arch == 'vgg16'):
        model = models.vgg16(pretrained = True)
    elif (model_arch == 'densenet121'):
        model = models.densenet121(pretrained = True)
    elif (model_arch == 'alexnet'):
        model = models.alexnet(pretrained = True)

    return model


def model_build(model, model_arch, drop_out):
    
    hidden_layers = args.hidden_layers
    ##num_categories = len(train_data.class_to_idx)
    num_categories = len(images_datasets['train'].class_to_idx)

    for param in model.parameters():
        param.requires_grad = False

    if (model_arch == 'vgg16'):
        from collections import OrderedDict
        classifier = nn.Sequential(OrderedDict([
                                  ('fc1', nn.Linear(25088, hidden_layers)),
                                  ('relu', nn.ReLU()),
                                  ('dropout', nn.Dropout(drop_out)),
                                  ('fc2', nn.Linear(hidden_layers,num_categories)),
                                  ('output', nn.LogSoftmax(dim=1))                             
                                  ]))
    elif (model_arch == 'densenet121'):
        from collections import OrderedDict
        classifier = nn.Sequential(OrderedDict([
                                  ('fc1', nn.Linear(1024,num_categories)),
                                  ('relu', nn.ReLU()),
                                  ('dropout', nn.Dropout(drop_out)),
                                  ('output', nn.LogSoftmax(dim=1))                             
                                  ]))
    elif (model_arch == 'alexnet'):
        from collections import OrderedDict
        classifier = nn.Sequential(OrderedDict([
                                  ('fc1', nn.Linear(9216, hidden_layers)),
                                  ('relu', nn.ReLU()),
                                  ('dropout', nn.Dropout(drop_out)),
                                  ('fc2', nn.Linear(hidden_layers,num_categories)),
                                  ('output', nn.LogSoftmax(dim=1))                             
                                  ]))
    else:
        print('error')
    
    return classifier


model = model_get(hyper_params['arch'].lower())
classifier_model = model_build(model, hyper_params['arch'].lower(), hyper_params['drop'])
model.classifier = classifier_model
print('\narch: ' + hyper_params['arch'] + ' classifier = ')
print(model.classifier)


def model_train(model, criterion, optimizer, epochs, train_loader, valid_loader, use_gpu):
    device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
    model.to(device)
    model.to(device)
    print_every = 20
    step = 0
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0
        
        for inputs, labels in train_loader:
            step += 1
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if step % print_every == 0:
                model.eval()
                accuracy = 0
                valid_loss = 0
                
                with torch.no_grad():
                    for inputs, labels in valid_loader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        outputs = model(inputs)
                        valid_loss += criterion(outputs, labels).item()
                        ps = torch.exp(outputs)
                        equality = (labels.data == ps.max(dim=1)[1])
                        accuracy += equality.type(torch.FloatTensor).mean()

                print(f"Epoch: {epoch+1}/{epochs}... "
                      f"Training Loss: {running_loss/print_every:.4f}... "
                      f"Validation Loss: {valid_loss/len(valid_loader):.4f}... "
                      f"Validation Accuracy: {accuracy/len(valid_loader):.4f}")
                
                running_loss = 0
                model.train()



def accu_check(model, test_load, gpu):    
    right = 0
    total = 0
    
    model.eval()
    
    if (gpu == 'GPU'):
        model.to('cuda:0')
    
    with torch.no_grad():
        for data in test_load:
            images, labels = data
            if (gpu == 'GPU'):
                images, labels = images.to('cuda'), labels.to('cuda')
            op = model(images)
            _, predicted = torch.max(op.data, 1)
            total += labels.size(0)
            right += (predicted == labels).sum().item()

    print('\nAccuracy: %d %%' % (100 * right / total))


criterion = nn.NLLLoss()
opti = optim.Adam(model.classifier.parameters(), hyper_params['lr'])
model_train(model, criterion, opti, hyper_params['epochs'], data_load['train_load'], data_load['valid_load'], hyper_params['gpu'])
accu_check(model, data_load['test_load'], hyper_params['gpu'])


model.class_to_idx = images_datasets['train'].class_to_idx
checkpoint = {
    'arch': hyper_params['arch'],
    'class_to_idx': model.class_to_idx, 
    'state_dict': model.state_dict(),
    'opti': opti.state_dict(),
    'in_layers': hyper_params['in_layers'],
    'hidden_layers': hyper_params['hidden_layers'],
    'out_layers': hyper_params['out_layers'],
    'learning rate': hyper_params['lr'],
    'dropout': hyper_params['drop'],
    'epochs': hyper_params['epochs'],
    'topk': hyper_params['topk']
}
torch.save(checkpoint, 'checkpoint.pth')
