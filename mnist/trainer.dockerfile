# Base image
FROM python:3.8-slim

# install python
RUN apt update && \
apt install --no-install-recommends -y build-essential gcc && \
apt clean && rm -rf /var/lib/apt/lists/*

# Copy essential parts from pc to container
COPY requirements.txt requirements.txt
COPY setup.py setup.py
COPY src/ src/
COPY data/ data/
COPY reports/ reports/
COPY models/ models/

# Set working directory of container
WORKDIR /
RUN pip install -r requirements.txt --no-cache-dir
# The cache is used to store away for future use. we want to disable to save space on our hard drive or to keep our Docker image as small as possible
# https://stackoverflow.com/questions/45594707/what-is-pips-no-cache-dir-good-for

ENTRYPOINT ["python", "-u", "src/models/train_model.py"]
# -u makes sure that any output from our script gets redicted to our console.


