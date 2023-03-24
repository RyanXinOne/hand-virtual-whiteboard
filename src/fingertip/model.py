import torch
import torch.nn as nn
from torchvision.models import vgg16


class FingertipDetector(nn.Module):
    '''Index fingertip detector model.

    Input: 128x128 RGB image
    Output: 2x1 vector of fingertip coordinates (x, y) in range [0, 1]
    '''

    def __init__(self, load_default_vgg_weights=True):
        super(FingertipDetector, self).__init__()
        self.vgg16 = vgg16(weights='DEFAULT' if load_default_vgg_weights else None)
        self.vgg16.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.vgg16(x)
        return x


def load_model(weights_path=None, device='cuda'):
    '''Load model weights from file or initialize weights if no path is given.

    Args:
        weights_path (str): Path to model weights file. If None, weights are initialized.
    '''
    model = FingertipDetector(load_default_vgg_weights=not weights_path).to(device)
    if not weights_path:
        # initialize weights
        for m in model.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
    else:
        model.load_state_dict(torch.load(weights_path, map_location=device))
        # print(f"Loaded weights from '{weights_path}'")
    return model


if __name__ == '__main__':
    import torchsummary
    model = load_model()
    torchsummary.summary(model, (3, 128, 128))
