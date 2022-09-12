"""
A Simple VGG-based Convolutional Neural Network for Sound Classification

Author: Benjamin Ho
Last updated: Sep 2022
"""
from torch import nn
from torchsummary import summary


class CNNNetwork(nn.Module):
    """
    A class definition for a VGG-based CNN for sound classification
    """
    def __init__(self, chs=[1, 16, 32, 64], out_ch=128, out_size=(5, 4)):
        """
        Constructor

        Args:
            chs (list, optional): Number of input channels at each convolution block. Defaults to [1, 16, 32, 64].
            out_ch (int, optional): _description_. Defaults to 128.
            out_size (tuple, optional): _description_. Defaults to (5, 4).
        """
        super().__init__()
        # 4 conv blocks -> flatten -> linear -> softmax
        self.conv1 = nn.Sequential(
            nn.Conv2d(chs[0], chs[1], kernel_size=3, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(chs[1], chs[2], kernel_size=3, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(chs[2], chs[3], kernel_size=3, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(chs[3], out_ch, kernel_size=3, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.flatten = nn.Flatten()
        self.adaptive_pool = nn.AdaptiveAvgPool2d(out_size)
        self.linear = nn.Linear(out_ch * out_size[0] * out_size[1], 10)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, input):
        x = self.conv1(input)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.adaptive_pool(x)
        x = self.flatten(x)
        logits = self.linear(x)
        predictions = self.softmax(logits)
        return predictions


if __name__ == "__main__":
    cnn = CNNNetwork()
    summary(cnn, (1, 64, 44))