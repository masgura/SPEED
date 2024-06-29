# SPEED
This repository contains the code to preprocess the spaceraft imagery dataset SPEED and train on it a detection neural network and a regression neural network.
First of all, download the dataset:
```
wget https://stacks.stanford.edu/file/druid:dz692fn7184/speed.zip
unzip speed.zip
```

Then launch

```
python3 partition.py
```
To apply train/dev/test split to the dataset and
```
python3 labels.py
```
To create labels for both detection and regression network.

To the training, testing and export to OpenVINO format the passages in the two Jupyter Notebook can be followed.
