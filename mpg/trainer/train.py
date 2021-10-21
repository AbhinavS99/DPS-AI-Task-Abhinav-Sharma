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


dataset_path = keras.utils.get_file("auto-mpg.data", "http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data")

column_names = ['MPG','Cylinders','Displacement','Horsepower','Weight',
                'Acceleration', 'Model Year', 'Origin']

dataset = pd.read_csv(dataset_path, names=column_names,
                      na_values = "?", comment='\t',
                      sep=" ", skipinitialspace=True)


# Dropping missing values
dataset = dataset.dropna()

# Origin is Categorical, thus creating dummy variables
dataset['Origin'] = dataset['Origin'].map({1: 'USA', 2: 'Europe', 3: 'Japan'})

dataset = pd.get_dummies(dataset, prefix='', prefix_sep='')

# Creating train-test split
train_dataset = dataset.sample(frac=0.8,random_state=0)
test_dataset = dataset.drop(train_dataset.index)

# Inspecting stats of train split
train_stats = train_dataset.describe()
train_stats.pop("MPG")
train_stats = train_stats.transpose()

def norm(x):
    return (x - train_stats['mean']) / train_stats['std']

train_labels = train_dataset.pop('MPG')
test_labels = test_dataset.pop('MPG')


# Normalize the train and the test dataset
normed_train_data = norm(train_dataset)
normed_test_data = norm(test_dataset)


def build_model():
   model = keras.Sequential([
     layers.Dense(64, activation='relu', input_shape=[len(train_dataset.keys())]),
     layers.Dense(32, activation='relu'),
     layers.Dense(16, activation='relu'),
     layers.Dense(4, activation='relu'),
     layers.Dense(1)
   ])

   optimizer = tf.keras.optimizers.RMSprop(0.001)

   model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae', 'mse'])
   return model

model = build_model()

EPOCHS = 1000

# The patience parameter is the amount of epochs to check for improvement
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

early_history = model.fit(normed_train_data, train_labels, 
                    epochs=EPOCHS, validation_split = 0.2, 
                    callbacks=[early_stop])


# Export model and save to GCS
model.save(BUCKET + '/mpg/model')
