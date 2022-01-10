import torch
from torch.utils.data import DataLoader, TensorDataset
from tests import _PATH_DATA
data_filepath = _PATH_DATA + "/data/processed"
bs = 1
n_classes = 10
train_images = torch.load(data_filepath + "/train_images.pt")
train_labels = torch.load(data_filepath + "/train_labels.pt")
train_set = TensorDataset(train_images, train_labels)
trainloader = DataLoader(train_set, batch_size=bs, shuffle=True)

test_images = torch.load(data_filepath + "/test_images.pt")
test_labels = torch.load(data_filepath + "/test_labels.pt")
test_set = TensorDataset(test_images, test_labels)
testloader = DataLoader(test_set, batch_size=bs, shuffle=True)

assert len(train_set) == len(train_labels), "The length of the training data did not match the length of the labels"
assert len(test_set) == len(test_labels), "The length of the testing data did not match the length of the labels"
for images, labels in trainloader:
    assert images.shape == torch.Size([1, 28, 28]), "Image shape not correct"
for images, labels in testloader:
    assert images.shape == torch.Size([1, 28, 28]), "Image shape not correct"
assert all(train_labels.unique() == torch.tensor(range(n_classes))), "All classes are not represented in data"
assert all(test_labels.unique() == torch.tensor(range(n_classes))), "All classes are not represented in data"