import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import torch
from torch import nn
import torch.optim as optim
import matplotlib.pyplot as plt
from src.models.model import MyAwesomeModel
from torch.utils.data import TensorDataset, DataLoader


@click.command()
@click.argument('data_filepath', type=click.Path(exists=True))
@click.argument('vis_path', type=click.Path(exists=True))
def train(data_filepath, vis_path):
    bs = 64
    # Load model
    model = MyAwesomeModel()

    # Load data
    train_images = torch.load(data_filepath + "/train_images.pt")
    train_labels = torch.load(data_filepath + "/train_labels.pt")
    train_set = TensorDataset(train_images, train_labels)
    trainloader = DataLoader(train_set, batch_size=bs, shuffle=True)

    test_images = torch.load(data_filepath + "/test_images.pt")
    test_labels = torch.load(data_filepath + "/test_labels.pt")
    test_set = TensorDataset(test_images, test_labels)
    testloader = DataLoader(test_set, batch_size=bs, shuffle=True)

    epochs = 5
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    training_loss, validation_loss, validation_accuracy, ep = [], [], [], []
    for e in range(epochs):
        train_loss = 0.0
        val_loss = 0.0
        acc = 0.0
        model.train()
        for images, labels in trainloader:
            optimizer.zero_grad()
            log_ps = model(images)
            loss = criterion(log_ps, labels)
            loss.backward()
            optimizer.step()

            # Calculate loss
            train_loss += loss.item()
        else:
            with torch.no_grad():
                for images, labels in testloader:
                    out = model(images)
                    loss = criterion(out, labels)
                    val_loss += loss.item()

                    predicted = torch.argmax(out, 1)
                    acc += (predicted == labels).sum()

            accuracy = acc/(len(testloader)*bs)
            ep.append(e+1)
            training_loss.append(train_loss/(len(trainloader)*bs))
            validation_loss.append(val_loss/(len(testloader)*bs))
            validation_accuracy.append(accuracy.item()*100)
            print("#"*15, " EPOCH", str(e+1) + " ", "#"*15)

    # save loss curves
    plt.plot(ep, training_loss, label="Training")
    plt.plot(ep, validation_loss, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(vis_path + "/loss_curves.png")
    plt.close()

    # Save accuracy curve
    plt.plot(ep, validation_accuracy)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.savefig(vis_path + "/accuracy_plot.png")

    # Save model
    torch.save(model.state_dict(), 'models/model_checkpoint.pth')


if __name__ == "__main__":
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())
    train()
