# Multi-Variable 학습시킬 값을 CSV 파일로 로딩

# Loading TensorFlow
import tensorflow as tf

# Loading Numpy
import numpy as np

# Loading CSV File
xy = np.loadtxt('data_01_test_score.csv', delimiter=',', dtype=np.float32)
