"""
Script for testing inference speed on MiDaS models using Apple Silicon M1 CPU vs GPU (mps)

Author: Benjamin Ho
Last Updated: 25 May 2022
"""

import cv2
import json
import torch
import urllib.request

import torch.nn as nn
import matplotlib.pyplot as plt


class MidasModel(nn.Module):
    def __init__(self, config):
        super(MidasModel, self).__init__()
        self.config = config
        model_type = self.config["model_type"]

        self.device = torch.device("mps") if self.config["accelerate"] else torch.device("cpu")

        assert model_type in ["DPT_Large", "DPT_Hybrid", "MiDaS_small"], "Model type chosen is not valid"

        self.net = torch.hub.load("intel-isl/MiDaS", model_type)
        self.net.to(self.device)
        
        if self.config["mode"] == "eval":
            self.net.eval()
        
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        if model_type == "MiDaS_small":
            self.transform = midas_transforms.small_transform
        else:
            self.transform = midas_transforms.dpt_transform
    
    def forward(self, x):
        input = self.transform(x).to(self.device)
        return self.net(input)

if __name__ == "__main__":
    with open("config.json", "r") as file:
        cfg = json.load(file)
    
    midas = MidasModel(cfg)

    img = cv2.imread(cfg["filepath"])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    with torch.no_grad():
        prediction = midas(img)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
    
    if cfg["accelerate"]:
        output = prediction.cpu().numpy()
    else:
        output = prediction.numpy()
    
    plt.imshow(output)
    plt.show()


        