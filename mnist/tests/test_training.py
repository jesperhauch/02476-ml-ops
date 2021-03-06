import torch
from torch.utils.data import DataLoader, TensorDataset
from src.models.model import MyAwesomeModel
import torch.optim as optim
from torch import nn
from tests import _PATH_DATA, _PROJECT_ROOT

data_filepath = _PATH_DATA + "/processed"
bs = 1
n_classes = 10
train_images = torch.load(data_filepath + "/train_images.pt")
train_labels = torch.load(data_filepath + "/train_labels.pt")
train_set = TensorDataset(train_images, train_labels)
trainloader = DataLoader(train_set, batch_size=bs, shuffle=True)

model = MyAwesomeModel()
state_dict = torch.load(_PROJECT_ROOT + "/models/model_checkpoint.pth")
model.load_state_dict(state_dict)

epochs = 5
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

for e in range(epochs):
    train_loss = 0.0
    model.train()
    for images, labels in trainloader:
        optimizer.zero_grad()
        log_ps = model(images)
        loss = criterion(log_ps, labels)
        assert abs(torch.exp(log_ps).sum().item() - 1.0) < 1e-6, "Model output does not sum to one"
        loss.backward()
        optimizer.step()

        # Calculate loss
        train_loss += loss.item()
