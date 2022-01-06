import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import torch
from src.models.model import MyAwesomeModel
import glob
from PIL import Image
from src.models.helper import *

@click.command()
@click.argument('model_filepath', type=click.Path(exists=True))
@click.argument('image_filepath', type=click.Path(exists=True))
def predict(model_filepath, image_folder):
    model = torch.load(model_filepath)
    model.eval()

    image_path = glob.glob(image_folder + "/*.png")
    if len(image_path) == 0:
        image_path = glob.glob(image_folder + "/*.jpg")

    images = np.array([])
    for img in image_path:
        images = np.append(images, np.array(Image.open(img).convert("RGB")))
    
    images = torch.tensor(images)
    images = (images - images.mean())/images.std()


    out = model(images)

    predicted = torch.argmax(out, 1)
    return predicted
