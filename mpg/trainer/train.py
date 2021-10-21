import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras

from tensorflow.keras import layers
print(tf.__version__)

BUCKET = 'gs://abhinav-dps-ai-bucket'

url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'
col_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight',
                'Acceleration', 'Model Year', 'Origin']

# Getting the dataset
dataset = pd.read_csv(url, names=col_names, na_values='?', comment='\t', sep=' ', skipinitialspace=True)

# Dropping missing values
dataset = dataset.dropna()

# Origin is Categorical, thus creating dummy variables
dataset['Origin'] = dataset['Origin'].map({1: '1', 2: '2', 3: '3'})
dataset = pd.get_dummies(dataset, columns=['Origin'])

# Creating train-test split
train_dataset = dataset.sample(frac=0.8, random_state=0)
test_dataset = dataset.drop(train_dataset.index)


train_stats = train_dataset.describe()
train_stats.pop("MPG")
train_stats = train_stats.transpose()
train_stats

train_labels = train_dataset.pop('MPG')
test_labels = test_dataset.pop('MPG')

def norm(x):
    return (x - train_stats['mean'])/train_stats['std']

train_dataset = norm(train_dataset)
test_dataset = norm(test_dataset)

def build_model():
    # Architecture -> normalization_layer || Dense(32, Relu) || Dense(64, Relu) || Dense(32, Relu) || Dense(1)
    model = keras.Sequential([
          layers.Dense(32, activation='relu'),
          layers.Dense(64, activation='relu'),
          layers.Dense(32, activation='relu'),
          layers.Dense(1)
    ])
    
    model.compile(loss='mean_absolute_error',
        optimizer=tf.keras.optimizers.Adam(0.001)
    )
    
    return model

model = build_model()
EPOCHS = 100

early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

history = model.fit(
    train_dataset,
    train_labels,
    validation_split=0.2,
    verbose=1, 
    epochs=100,
    callbacks=[early_stop]
)
model.save(BUCKET + '/mpg/model')
