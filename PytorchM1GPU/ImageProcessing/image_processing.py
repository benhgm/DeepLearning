import cv2
import time
import torch
import argparse
import numpy as np
import torchvision.transforms as transforms


class ImageProcessing(object):
    def __init__(self, args):
        self.transforms = transforms.Compose([
            transforms.CenterCrop((192, 640)),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        self.device = torch.device(args.device)
        self.image_path = args.image_path
    
    def run(self):
        img = cv2.imread(self.image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.float32(img / 255.0)
        img_tensor = torch.from_numpy(img)
        img_tensor = torch.permute(img_tensor, (2, 0, 1))
        img_tensor.to(self.device)
        start = time.time()
        transformed_image = self.transforms(img_tensor)
        end = time.time()
        print("Time taken: {}s".format(end-start))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Basic cleaning of dataset")

    parser.add_argument(
        "--device",
        type=str,
        help="device to send data to for processing",
        default="cpu"
    )

    parser.add_argument(
        "--image_path",
        type=str,
        help="path to the image file",
        required=True
    )
    
    args = parser.parse_args()

    image_processing = ImageProcessing(args)

    image_processing.run()