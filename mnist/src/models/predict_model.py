import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import torch
from torch.utils.data import TensorDataset, DataLoader
from src.models.model import MyAwesomeModel
from src.models import helper
import matplotlib.pyplot as plt

@click.command()
@click.argument('data_filepath', type=click.Path(exists=True))
@click.argument('model_filepath', type=click.Path(exists=True))
@click.argument('predict_filepath', type=click.Path(exists=True))
def predict(data_filepath, model_filepath, predict_filepath):
    model = MyAwesomeModel()

    state_dict = torch.load(model_filepath)

    model.load_state_dict(state_dict)
    bs = 1

    train_images = torch.load(data_filepath + "/train_images.pt")
    train_labels = torch.load(data_filepath + "/train_labels.pt")
    train_set = TensorDataset(train_images, train_labels)
    trainloader = DataLoader(train_set, batch_size=bs, shuffle=True)

    # Get out image and labels
    data = iter(trainloader)
    images, _ = next(data)
    img = images[0]
    
    # Convert 2D image to 1D vector
    img = img.view(1, 784)
    with torch.no_grad():
        ps = torch.exp(model.forward(img))
    helper.view_classify(img.view(1, 28, 28), ps, version="MNIST")
    plt.savefig(predict_filepath + "/predicted.png")

if __name__ == "__main__":
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())
    predict()