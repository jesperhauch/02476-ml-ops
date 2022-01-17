"""
LFW dataloading
"""
import argparse
import time

import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.utils import make_grid
import torchvision.transforms.functional as F
import glob
import os
import csv

class LFWDataset(Dataset):
    def __init__(self, path_to_folder: str, transform) -> None:
        self.image_paths = glob.glob(path_to_folder + '**/*.jpg', recursive=True)
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, index: int) -> torch.Tensor:
        img_idx = self.image_paths[index]
        img = Image.open(img_idx).convert("RGB")
        return self.transform(img)

def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    fix.savefig("pic_grid.jpg")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-path_to_folder', default='lfw-deepfunneled/', type=str)
    parser.add_argument('-batch_size', default=8, type=int)
    parser.add_argument('-num_workers', default=4, type=int)
    parser.add_argument('-visualize_batch', action='store_true')
    parser.add_argument('-get_timing', action='store_true')
    parser.add_argument('-error_barplot', action='store_true')
    parser.add_argument('-batches_to_check', default=100, type=int)
    
    args = parser.parse_args()
    
    lfw_trans = transforms.Compose([
        transforms.RandomAffine(5, (0.1, 0.1), (0.5, 2.0)),
        transforms.ToTensor()
    ])
    
    # Define dataset
    dataset = LFWDataset(args.path_to_folder, lfw_trans)
    print(dataset.__len__())
    # Define dataloader
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=args.num_workers
    )
    
    if args.visualize_batch:
        images = next(iter(dataloader))
        grid = make_grid(images)
        show(grid)
        
    if args.get_timing:
        # lets do some repetitions
        with open("timing.txt", "a") as f:
            res = [ ]
            for _ in range(5):
                start = time.time()
                for batch_idx, batch in enumerate(dataloader):
                    if batch_idx > args.batches_to_check:
                        break
                end = time.time()

                res.append(end - start)
            
            res = np.array(res)
            f.write(str(args.num_workers) + "\t" + str(np.mean(res)) + "\t" + str(np.std(res)) + "\n")
            print(f'Timing: {np.mean(res)}+-{np.std(res)}')
        f.close()

    if args.error_barplot:
        with open("timing.txt") as f:
            runs = csv.reader(f, delimiter="\t")
            num_workers = []
            avgs = []
            stds = []
            for run in runs:
                num_workers.append(float(run[0]))
                avgs.append(float(run[1]))
                stds.append(float(run[2]))
        f.close()
        fig, ax = plt.subplots()
        ax.errorbar(x=num_workers, y=avgs, yerr=stds)
        ax.set_ylabel("Time")
        ax.set_xlabel("Number of workers")
        fig.savefig("error_plot.png")
