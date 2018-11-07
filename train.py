import torch
import time
import copy
import argparse
from torch import nn,optim
from collections import OrderedDict
from torch.optim import lr_scheduler
from torchvision import datasets, transforms, models


arch = {"vgg16":25088,
        "resnet18":512}

def load_data(where = './flowers'):

    data_dir = where
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomRotation(30),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'valid': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    dirs = {'train': train_dir,
            'valid': valid_dir,
            'test': test_dir}

    image_datasets = {x: datasets.ImageFolder(dirs[x], transform=data_transforms[x]) for x in
                      ['train', 'valid', 'test']}

    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=32, shuffle=True) for x in
                   ['train', 'valid', 'test']}

    # check the datasize
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'valid', 'test']}

    return dataloaders, dataset_sizes, image_datasets

def build_model(arch, hidden_units):

    archs = {'vgg16':25088,
             'resnet18':512}

    if arch == 'vgg16':
        model = models.vgg16(pretrained=True)
    elif arch == 'resnet18':
        model = models.resnet18(pretrained=True)
    else:
        print('The pre-trained model arch should be {} or {}'.format('vgg16','resnet18'))

    for param in model.parameters():
        param.requires_grad = False


    dropout_rate = 0.5
    classifier = nn.Sequential(OrderedDict([
                              ('fc1', nn.Linear(archs[arch], hidden_units)),
                              ('relu', nn.ReLU()),
                              ('fc2', nn.Linear(hidden_units, 102)),
                              ('drop1', nn.Dropout(p=dropout_rate)),
                              ('output', nn.LogSoftmax(dim=1))]))
    model.classifier = classifier
    return model

def train_model(model, learning_rate, num_epochs=25, gpu=True):

    if gpu:
        device = 'cuda'
    else:
        device = 'cpu'


    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.1)
    # Number of epochs
    eps = num_epochs

    dataloaders, dataset_sizes, image_datasets = load_data(args.data_dir)

    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(eps):
        print('Epoch {}/{}'.format(epoch, eps - 1))
        print('-' * 10)

        # Each epoc nh has a training and validation phase
        for phase in ['train', 'valid']:
            if phase == 'train':
                # different methods for adjusting the learning rate and step size used during optimization
                scheduler.step()
                # Set model to training mode
                model.train()
            else:
                # Set model to evaluate mode
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    #  get the image that is most likely
                    _, preds = torch.max(outputs, 1)
                    # method used to evaluate the model fit.
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        # optimization technique used to update the weights
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)

    return model

def save_checkpoint(model,image_datasets,arch):

    model.class_to_idx = image_datasets['train'].class_to_idx
    model.cpu()
    torch.save({'arch': '{}'.format(arch),
                'state_dict': model.state_dict(),
                'class_to_idx': model.class_to_idx},
                'classifier_{}.pth'.format(arch))

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('data_dir', nargs='*', action="store", default="./flowers/")


    # training validation log(training loss, validation loss, validation accuracy)
    # Model architecture(at least two different architecture)
    parser.add_argument("--arch", required=False, default='vgg16',
                    help='Pre-trained model from trochvision vgg16 or resnet18')
    

    # Model hyperparameters(learning rate, number of hidden units, training epochs)
    parser.add_argument('--learning_rate',required=False, default=0.01, help='learning rate')
    parser.add_argument("--hidden_units", required=False, default= 4096, help='hidden units for fc layer')
    parser.add_argument('--epochs',required=False,default=10, type=int, help = 'number of training epochs')

    # Training with GPU()
    parser.add_argument('--gpu', type=bool, default=False, help='Use GPU or not')

    parser.add_argument('--saved_model', type=str, default='my_checkpoint_cmd.pth', help='path of your saved model')

    args = parser.parse_args()

    print(args.data_dir)
    print(args.arch == 'vgg16')

    dataloaders, dataset_sizes, image_datasets = load_data(args.data_dir)

    model = build_model(args.arch,args.hidden_units)
    model_ft = train_model(model,args.learning_rate,args.epochs,args.gpu)

    checkpoint = save_checkpoint(model_ft,image_datasets=image_datasets, arch=args.arch)

    print(checkpoint)