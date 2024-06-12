import os
import shutil
import pandas as pd

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
import keras

import matplotlib.pyplot as plt


print("TF Version: ", keras.__version__)



tf.get_logger().setLevel('ERROR')

