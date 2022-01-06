from src.models.helper import *
import torch
import click
from torch.utils.data import TensorDataset, DataLoader


@click.command()
@click.argument('model_filepath', type=click.Path(exists=True))
@click.argument('data_filepath', type=click.Path(exists=True))
def visualize(model_filepath, data_filepath):
    # Implement evaluation logic here
        model = torch.load(model_filepath)
        model.eval()
        bs = 64

        # Load data
        train_images = torch.load(data_filepath + "/train_images.pt")
        train_labels = torch.load(data_filepath + "/train_labels.pt")
        train_set = TensorDataset(train_images, train_labels)
        trainloader = DataLoader(train_set, batch_size=bs, shuffle=True)