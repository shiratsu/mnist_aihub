from __future__ import print_function

import glob
import math
import os

# from IPython import display
from matplotlib import cm
from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import metrics
# tensorflow_version 1.x
import tensorflow as tf
from tensorflow.python.data import Dataset

# labelや特徴をパースする
def parse_labels_and_features(dataset):
  """Extracts labels and features.

  This is a good place to scale or transform the features if needed.

  Args:
    dataset: A Pandas `Dataframe`, containing the label on the first column and
      monochrome pixel values on the remaining columns, in row major order.
  Returns:
    A `tuple` `(labels, features)`:
      labels: A Pandas `Series`.
      features: A Pandas `DataFrame`.
  """

  # これがなんなのかちゃんとはっきり確認した方が良い

  labels = dataset[0]

  # DataFrame.loc index ranges are inclusive at both ends.
  # １列目から最後まで
  features = dataset.loc[:,1:784]
  # Scale the data to [0, 1] by dividing out the max value, 255.
  features = features / 255

  return labels, features

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format

mnist_dataframe = pd.read_csv(
  "https://download.mlcc.google.com/mledu-datasets/mnist_train_small.csv",
  sep=",",
  header=None)

# Use just the first 10,000 records for training/validation.
mnist_dataframe = mnist_dataframe.head(10000)

mnist_dataframe = mnist_dataframe.reindex(np.random.permutation(mnist_dataframe.index))
print(mnist_dataframe.shape)

# 72列目と73列目を出力
print("72列目と73列目を出力:")
print(mnist_dataframe.loc[:, 73:750].shape)
# 0列目を出力
print("0列目:")
print(mnist_dataframe.loc[:, 0:0].shape)

# ゼロ行目はなんだ
print("what is at 0 row:")
print(mnist_dataframe[0].shape)# 一番最初の列のことらしい
print(mnist_dataframe.loc[0].shape)# 列を行として出力

# トレーニング用のデータがこちら
training_targets, training_examples = parse_labels_and_features(mnist_dataframe[:7500])
print(training_examples.describe())

# 検証用のデータがこちら
validation_targets, validation_examples = parse_labels_and_features(mnist_dataframe[7500:10000])
validation_examples.describe()

# show random example
rand_example = np.random.choice(training_examples.index)
_, ax = plt.subplots()
ax.matshow(training_examples.loc[rand_example].values.reshape(28, 28))
ax.set_title("Label: %i" % training_targets.loc[rand_example])
ax.grid(False)
