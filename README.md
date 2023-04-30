# CS-5830/6830 Final Project

_**Group members:** Adam Alder, Kevin Vulcano, Brandon Kolesar_

## Datasets

This project used two datasets, both from Kaggle.

* Aircraft images dataset: https://www.kaggle.com/datasets/seryouxblaster764/fgvc-aircraft
* Bird images dataset: https://www.kaggle.com/datasets/gpiosenka/100-bird-species

## Repository Organization

There are three Jupyter Notebooks:

* `preprocessing.ipynb` - Contains the code to preprocess the images.

* `final-project.ipynb` - Contains the code to train the CNN models and report on their accuracy.

* `visualizations.ipynb` - Contains the code to visualize the CNN.

The CNN models are defined in the `models/CNN_model.py` and `models/CNN_SE_model.py` files.

### Branches

Due to performing the same analysis three times, once on only bird images, once on only aircraft images, and once on a combination of bird and aircraft images, the code and results were split up into different Git branches.

* `birds-only` - The code modified to only work on bird images and its results.

* `aircrafts-only` (not `aircraft-only`) - The code modified to only work on aircraft images and its results.

* `combined-data` - The code unmodified and its results. Works on both bird and aircraft images.