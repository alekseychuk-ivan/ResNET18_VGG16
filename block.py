import torch
from torchvision import models

class ResnetBlock(torch.nn.Module):
    def __init__(self, in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, use_conv1_1=None):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                     stride=stride, padding=padding)
        self.conv2 = torch.nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size,
                                     padding=padding)
        if use_conv1_1:
            self.conv1_1 = torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1,
                                           stride=stride)
        else:
            self.conv1_1 = None

        self.bn1 = torch.nn.BatchNorm2d(out_channels)
        self.bn2 = torch.nn.BatchNorm2d(out_channels)

    def forward(self, X):
        out = torch.nn.ReLU()(self.bn1(self.conv1(X)))
        out = self.bn2(self.conv2(out))
        if self.conv1_1:
            out += self.conv1_1(X)
        else:
            out += X
        return torch.nn.ReLU()(out)


class VGGBlock(torch.nn.Module):
    def __init__(self, ):
        super().__init__()
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1, stride=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=1),
            torch.nn.ReLU(),
            )
        self.pool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=64, out_channels=128, padding=1, stride=1, kernel_size=3),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=128, out_channels=128, padding=1, stride=1, kernel_size=3),
            torch.nn.ReLU(),
            )
        self.pool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=128, out_channels=256, padding=1, kernel_size=3, stride=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, stride=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, stride=1),
            torch.nn.ReLU(),
            )
        self.pool3 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer4 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=256, out_channels=512, padding=1, kernel_size=3, stride=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=1),
            torch.nn.ReLU(),
            )
        self.pool4 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer5 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=512, out_channels=512, padding=1, kernel_size=3, stride=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=1),
            torch.nn.ReLU(),
            )
        self.pool5 = torch.nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        out = self.layer1(x)
        out = self.pool1(out)
        out = self.layer2(out)
        out = self.pool2(out)
        out = self.layer3(out)
        out = self.pool3(out)
        out = self.layer4(out)
        out = self.pool4(out)
        out = self.layer5(out)
        out = self.pool5(out)
        return out

def handle_create(net_name, num_classes):
    if net_name == 'ResNET':
        b1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3, out_channels=64, stride=2, padding=3, kernel_size=7),
            torch.nn.BatchNorm2d(64),
            torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        b2 = ResnetBlock(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, use_conv1_1=None)
        b3 = ResnetBlock(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, use_conv1_1=None)
        b4 = ResnetBlock(in_channels=64, out_channels=128, kernel_size=3, stride=2, use_conv1_1=True)
        b5 = ResnetBlock(in_channels=128, out_channels=128, kernel_size=3, stride=1, use_conv1_1=None)
        b6 = ResnetBlock(in_channels=128, out_channels=256, kernel_size=3, stride=2, use_conv1_1=True)
        b7 = ResnetBlock(in_channels=256, out_channels=256, kernel_size=3, stride=1, use_conv1_1=None)
        b8 = ResnetBlock(in_channels=256, out_channels=512, kernel_size=3, stride=2, use_conv1_1=True)
        b9 = ResnetBlock(in_channels=512, out_channels=512, kernel_size=3, stride=1, use_conv1_1=None)

        return torch.nn.Sequential(b1, b2, b3, b4, b5, b6, b7, b8, b9,
                                      torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(),
                                      torch.nn.Linear(512, num_classes))


    elif net_name == 'VGG':
        model = VGGBlock()
        classifier = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(25088, 4096), torch.nn.ReLU(), torch.nn.Dropout(0.5),
            torch.nn.Linear(4096, 4096), torch.nn.ReLU(), torch.nn.Dropout(0.5),
            torch.nn.Linear(4096, num_classes))
        return torch.nn.Sequential(model, classifier)


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def initialize_model(model_name, num_classes=2, feature_extract=True, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None

    if model_name == "ResNET":
        """ Resnet18
        """
        model_ft = models.resnet18(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = torch.nn.Linear(num_ftrs, num_classes)

    elif model_name == "VGG":
        """ VGG16
        """
        model_ft = models.vgg16(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = torch.nn.Linear(num_ftrs, num_classes)

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft
