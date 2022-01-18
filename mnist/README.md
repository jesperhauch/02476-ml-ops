mnist
==============================

**M5 Git**\
I've used `git` in other courses and for work, so I feel like I already had the hang of this.

**M6 Code Structure**\
I used `cookiecutter` to generate the structure of this folder as indicated by the Project Organization chart below.
I love having a standardized structure to work in and I will definitely be using it in future projects. I think it is a major advantage in terms of reproduceability.

**M7 Good Coding Practice**\
I haven't ever though of `pep8` compliance when writing code previously. It's something I'll have in mind, but I especially appreciated the autolinting Nicki showcased during the lecture.

**M8 Data Version Control**\
I have yet to experience the advantages of using DVC. I've found it to be rather unstable during our project work and most of my projects concern a static dataset.
It is definitely something I'll have to explore further.

**M9 Docker**\
I think Docker has been a really difficult subject for me to understand fully. It was good experience to write the `.dockerfile` though and I'm sure I'll see more to Docker in the future.

**M15 Continuous Integration**\
It was pretty interesting to generate coverage and write the unit tests on the code. In this session, I had a lot of issues getting the Github Actions workflows to work. I got the workflow with `isort` up and running, but the others are still failing to this day.

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
