import os
import torch
import torchvision
from torchvision import transforms
import time
from block import handle_create, initialize_model
import argparse

num_classes = 2  # num classes
BATCH_SIZE = 8  #
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')  # choice device

# parse command from terminal
parse = argparse.ArgumentParser()
parse.add_argument('--datapath', '-dp',
                   help='Top level data directory. Here we assume the format of the directory conforms to the ImageFolder structure. Default input/hymenoptera_data/val ',
                   default='input/hymenoptera_data/val')
parse.add_argument('--checkpoint', '-cp', help='Path to models checkpoint', default='./output')
parse.add_argument('--model', '-m', help='What model will be eval, ResNET or VGG?', default='ResNET')
parse.add_argument('--feature_extract', '-fe',
                   help='Flag for feature extracting. When False, we finetune the whole model, when True we only update the reshaped layer params',
                   default=True)
parse.add_argument('--use_pretrained', '-up', help='Do you want use pretrain weight, True or False?', default=True)

args = parse.parse_args()
net_name = args.model
feature_extract = args.feature_extract
use_pretrained = args.use_pretrained
datapath = args.datapath
saveto = args.checkpoint

# transform input image
transformer = torchvision.transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

transformer_val = torchvision.transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


# function for eval model
def eval_model(model, dataloaders, name=False):
    since = time.time()
    if name:
        print('Validate {} wait please'.format(name))

    model.eval()  # Set model to evaluate mode

    running_corrects = 0

    # Iterate over data.
    for inputs, labels in dataloaders:
        inputs = inputs.to(DEVICE)
        labels = labels.to(DEVICE)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        running_corrects += torch.sum(preds == labels.data)
    best_acc = running_corrects.double() / len(dataloaders.dataset)
    time_elapsed = time.time() - since
    print('Validation complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60,))
    print('Validation Acc {}: {:4f}\n'.format(name, best_acc))


flag = True
lst_dir = os.listdir(saveto)  # list file in checkpoint path
# check validation folder
if not os.path.exists(datapath + '/val'):
    print('The val or train folder does not exist at the specified path.')
    flag = False
    print('Script close')
    exit()

# choice net and create model
if net_name.lower() == 'ResNET'.lower():
    name_lst = ['ResNET18', 'ResNET18_FT', 'ResNET18_AUG']
    lst_pth = ['ResNET18.pth', 'ResNET18_FT.pth', 'ResNET18_AUG.pth']

    if set(lst_pth).issubset(set(lst_dir)):
        model_h = handle_create('ResNET', num_classes=num_classes)
        model_f = initialize_model('ResNET', num_classes=num_classes, feature_extract=feature_extract,
                                   use_pretrained=True)
        model_aug = initialize_model('ResNET', num_classes=num_classes, feature_extract=feature_extract,
                                     use_pretrained=True)
    else:
        print('Please check the files {}'.format(lst_pth))
        flag = False
        print('Script close')
        exit()
elif net_name.lower() == 'VGG'.lower():
    name_lst = ['VGG16', 'VGG16_FT', 'VGG16_AUG']
    lst_pth = ['VGG16.pth', 'VGG16_FT.pth', 'VGG16_AUG.pth']
    if set(lst_pth).issubset(set(lst_dir)):
        model_h = handle_create('VGG', num_classes=num_classes)
        model_f = initialize_model('VGG', num_classes=num_classes, feature_extract=feature_extract, use_pretrained=True)
        model_aug = initialize_model('VGG', num_classes=num_classes, feature_extract=feature_extract,
                                     use_pretrained=True)
    else:
        print('Please check the files {}'.format(lst_pth))
        flag = False
        print('Script close')
        exit()

else:
    print('Such a model does not exist')
    print('Script close')
    flag = False

# eval model
if flag:
    model_lst = [model_h, model_f, model_aug]
    for model, param in zip(model_lst, lst_pth):
        check = torch.load('{}/{}'.format(saveto, param))
        model.load_state_dict(torch.load('{}/{}'.format(saveto, param)))

    test_data = torchvision.datasets.ImageFolder("{}".format(datapath), transform=transformer)
    test_aug = torchvision.datasets.ImageFolder("{}".format(datapath), transform=transformer)

    val_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=False)
    val_loader_aug = torch.utils.data.DataLoader(dataset=test_aug, batch_size=BATCH_SIZE, shuffle=False)
# preprocess data for eval_model
#     data_dict = {'val': val_loader}
#     data_aug = {'val': val_loader_aug}
    data_lst = [val_loader, val_loader, val_loader_aug]
    for model, data, name in zip(model_lst, data_lst, name_lst):
        for param in model.parameters():
            param.requires_grad = False
        torch.cuda.empty_cache()
        model = model.to(DEVICE)
        eval_model(model=model, dataloaders=data, name=name)
