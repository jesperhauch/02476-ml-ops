import sys
sys.path.append(".")
from torchvision import models
from torchvision import transforms
from distributed_data_loading.lfw_dataset import LFWDataset
from torch.utils.data import DataLoader
import time
resnet = models.resnet152(pretrained=True)
mobilenet = models.mobilenet_v3_large(pretrained=True)
# Define dataset
trs = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0,0,0], std=[1,1,1])])
dataset = LFWDataset("/home/jesperhauch/02476-ml-ops/s7_scalable/distributed_data_loading/lfw-deepfunneled", trs)
# Define dataloader
dataloader = DataLoader(
    dataset, 
    batch_size=4, 
    shuffle=False,
    num_workers=4
)

img = next(iter(dataloader))
start_resnet = time.time()
resnet.forward(img)
end_resnet = time.time()
print("ResNet:", end_resnet-start_resnet)
start_mobilenet = time.time()
mobilenet.forward(img)
end_mobilenet = time.time()
print("MobileNet:", end_mobilenet-start_mobilenet)


