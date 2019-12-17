import torch
import torch.nn as nn


class AlexNetSelf(nn.Module):

    def __init__(self):
        super(AlexNetSelf, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=5),
            nn.BatchNorm2d(64,track_running_stats=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2,padding=1))
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.BatchNorm2d(192,track_running_stats=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        self.conv3 = nn.Sequential(
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.BatchNorm2d(384,track_running_stats=False),
            nn.ReLU(inplace=True))
        self.conv4 =  nn.Sequential(
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256,track_running_stats=False),
            nn.ReLU(inplace=True))
        self.conv5 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256,track_running_stats=False),
            nn.ReLU(inplace=True))
        self.avgpool = nn.AdaptiveAvgPool2d((4, 4))

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x


class BinClassifier(nn.Module):
    def __init__(self):
        super(BinClassifier, self).__init__()
        self.lin = nn.Sequential(
            nn.Linear(256 * 4 * 4 * 2, 4096),
            nn.BatchNorm1d(4096, track_running_stats=False),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 2),
        )

    def forward(self, x):
        #print(x.size())
        x = self.lin(x)
        return x


class RotClassifier(nn.Module):
    def __init__(self):
        super(RotClassifier, self).__init__()
        self.lin = nn.Sequential(
            #nn.Dropout(0.7),
            nn.Linear(256 * 4 * 4, 1024),
            nn.BatchNorm1d(1024,track_running_stats=False),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 4),
        )

    def forward(self, x):
        x=self.lin(x)
        return x


class Normal_test(nn.Module):
    def __init__(self, num_classes):
        super(Normal_test, self).__init__()
        self.classifier = nn.Sequential(
            nn.Dropout(0.7),
            nn.Linear(256 * 4 * 4, 4096),
            nn.BatchNorm1d(4096,track_running_stats=False),
            nn.ReLU(inplace=True),
            nn.Dropout(0.7),
            nn.Linear(4096, 4096),
            nn.BatchNorm1d(4096,track_running_stats=False),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x=self.classifier(x)
        return x


class NormalClassifier(nn.Module):
    def __init__(self, num_classes):
        super(NormalClassifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Dropout(0.7),
            nn.Linear(256 * 4 * 4, 4096),
            nn.BatchNorm1d(4096,track_running_stats=False),
            nn.ReLU(inplace=True),
            nn.Dropout(0.7),
            nn.Linear(4096, 4096),
            nn.BatchNorm1d(4096,track_running_stats=False),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x=self.classifier(x)
        return x


class NormalClassifierShort(nn.Module):
    def __init__(self, num_classes):
        super(NormalClassifierShort, self).__init__()
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 4 * 4, num_classes),
            #nn.BatchNorm1d(4096),
            #nn.ReLU(inplace=True),
            #nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x=self.classifier(x)
        return x

class AlexNet(nn.Module):

    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
