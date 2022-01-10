import sys
sys.path.append(".")
import torch
from torch.utils.data import DataLoader, TensorDataset
from src.models.model import MyAwesomeModel
import pytest
from tests import _PATH_DATA, _PROJECT_ROOT

data_filepath = _PATH_DATA + "/data/processed"
bs = 1
n_classes = 10
train_images = torch.load(data_filepath + "/train_images.pt")
train_labels = torch.load(data_filepath + "/train_labels.pt")
train_set = TensorDataset(train_images, train_labels)
trainloader = DataLoader(train_set, batch_size=bs, shuffle=True)

model = MyAwesomeModel()
state_dict = torch.load(_PROJECT_ROOT + "/models/model_checkpoint.pth")
model.load_state_dict(state_dict)
for images, _ in trainloader:
    assert images.shape == torch.Size([1, 28, 28])
    output = model.forward(images)
    assert output.shape == torch.Size([1, 10])
    assert abs(torch.exp(output).sum().item() - 1.0) < 1e-6