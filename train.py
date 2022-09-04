import os
import torch
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
import time
import copy
from block import handle_create, initialize_model
import argparse

# parse command from terminal
parse = argparse.ArgumentParser()
parse.add_argument('--datapath', '-dp',
                   help='Top level data directory. Here we assume the format of the directory conforms to the ImageFolder structure. Default input/hymenoptera_data/train',
                   default='input/hymenoptera_data/train')
parse.add_argument('--saveto', '-st', help='Path to save checkpoint', default='./output')
parse.add_argument('--model', '-m', help='What model will be train, ResNET or VGG?', default='ResNET')
parse.add_argument('--feature_extract', '-fe',
                   help='Flag for feature extracting. When False, we finetune the whole model, when True we only update the reshaped layer params',
                   default=True)
parse.add_argument('--use_pretrained', '-up', help='Do you want use pretrain weight, True or False?', default=True)
args = parse.parse_args()

net_name = args.model
feature_extract = args.feature_extract
use_pretrained = args.use_pretrained
datapath = args.datapath
saveto = args.saveto

# check train folder and folder for save parameters
if not os.path.exists(datapath):
    print('The train folder does not exist at the specified path.')
    exit()
if not os.path.exists(saveto):
    os.mkdir(saveto)


DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')  # choice device
# DEVICE = torch.device('cpu')  # uncomment this if problem with cuda
BATCH_SIZE = 8
num_classes = 2  # num output classes
torch.cuda.empty_cache()  # clear cuda cache

# transform input images
transformer = torchvision.transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

transformer_train = torchvision.transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

transformer_val = torchvision.transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
# load and split data
train = torchvision.datasets.ImageFolder("{}".format(datapath), transform=transformer)
trainAug = torchvision.datasets.ImageFolder("{}".format(datapath), transform=transformer_train)
train_data, test_data = torch.utils.data.random_split(train, [int(len(train) * 0.8), len(train) - int(0.8 * len(train))])
# test_data = torchvision.datasets.ImageFolder("{}/val".format(datapath), transform=transformer)
train_aug, test_aug = torch.utils.data.random_split(trainAug, [int(len(trainAug) * 0.8), len(trainAug) - int(0.8 * len(trainAug))])
# test_aug = torchvision.datasets.ImageFolder("{}/val".format(datapath), transform=transformer_val)

train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
val_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=False)

train_loader_aug = torch.utils.data.DataLoader(dataset=train_aug, batch_size=BATCH_SIZE, shuffle=True)
val_loader_aug = torch.utils.data.DataLoader(dataset=test_aug, batch_size=BATCH_SIZE, shuffle=False)

# image, label = train_data[15]
# plt.imshow(image.permute(2, 1, 0))
# plt.show()
# print(label, train_data.classes[label])

# train function
def train_model(model, dataloaders, criterion, optimizer, num_epochs=20, name=False):
    since = time.time()

    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    if name:
        print('Train {} wait please'.format(name))

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))


            # deep copy the model
            if phase == 'test' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'test':
                val_acc_history.append(epoch_acc)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60,))
    print('Best test Acc {}: {:4f}\n'.format(name, best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

# choice net and create models
if net_name.lower() == 'ResNET'.lower():
    model_h = handle_create('ResNET', num_classes=2)
    model_f = initialize_model('ResNET', num_classes=num_classes, feature_extract=feature_extract, use_pretrained=True)
    model_aug = initialize_model('ResNET', num_classes=num_classes, feature_extract=feature_extract, use_pretrained=True)
    name_lst = ['ResNET18', 'ResNET18_FT', 'ResNET18_AUG']
elif net_name.lower() == 'VGG'.lower():
    model_h = handle_create('VGG', num_classes=2)
    model_f = initialize_model('VGG', num_classes=num_classes, feature_extract=feature_extract, use_pretrained=True)
    model_aug = initialize_model('VGG', num_classes=num_classes, feature_extract=feature_extract, use_pretrained=True)
    name_lst = ['VGG16', 'VGG16_FT', 'VGG16_AUG']
else:
    print('Such a model does not exist')
    exit()

model_lst = [model_h, model_f, model_aug]  # model list

loss = torch.nn.CrossEntropyLoss()  # loss function

num_epochs = 20  # num epochs for train
# data for model train
data_dict = {'train': train_loader, 'test': val_loader}
data_aug = {'train': train_loader_aug, 'test': val_loader_aug}
data_lst = [data_dict, data_dict, data_aug]

#train model
for model, data, name in zip(model_lst, data_lst, name_lst):
    trainer = torch.optim.Adam(model.parameters(), lr=9e-5)
    torch.cuda.empty_cache()
    model = model.to(DEVICE)
    model = train_model(model=model, dataloaders=data, criterion=loss, optimizer=trainer, name=name)
    torch.cuda.empty_cache()
    model = model.to(torch.device('cpu'))
    torch.save(model.state_dict(), '{}/{}.pth'.format(saveto, name))
