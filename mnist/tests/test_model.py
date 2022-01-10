import sys
sys.path.append(".")
import torch
from torch.utils.data import DataLoader, TensorDataset
from src.models.model import MyAwesomeModel
import pytest

data_filepath = "data/processed"
bs = 1
n_classes = 10
train_images = torch.load(data_filepath + "/train_images.pt")
train_labels = torch.load(data_filepath + "/train_labels.pt")
train_set = TensorDataset(train_images, train_labels)
trainloader = DataLoader(train_set, batch_size=bs, shuffle=True)

model = MyAwesomeModel()
state_dict = torch.load("models/model_checkpoint.pth")
model.load_state_dict(state_dict)
def test_error_on_wrong_shape():
    with pytest.raises(ValueError, match = "Expected each sample to have shape [1, 28, 28]"):
        for images, _ in trainloader:
            assert images.shape == torch.Size([1, 28, 28])
            output = model.forward(images)
            assert output.shape == torch.Size([1, 10])
            assert abs(torch.exp(output).sum().item() - 1.0) < 1e-6