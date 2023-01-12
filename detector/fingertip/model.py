import torch.nn as nn
from torchvision.models import vgg16_bn
import torchsummary
from utils import device


class FingertipDetector(nn.Module):
    def __init__(self):
        super(FingertipDetector, self).__init__()
        self.vgg16 = vgg16_bn(weights='DEFAULT')
        self.vgg16.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 512),
            nn.ReLU(True),
            nn.Linear(512, 2),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.vgg16(x)
        return x


if __name__ == '__main__':
    model = FingertipDetector().to(device)
    torchsummary.summary(model, (3, 128, 128))
