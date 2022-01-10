import sys
sys.path.append(".")
import torch
from torch.utils.data import DataLoader, TensorDataset
from src.models.model import MyAwesomeModel
import torch.optim as optim
from torch import nn

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

epochs = 5
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

training_loss, validation_loss, validation_accuracy, ep = [], [], [], []
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