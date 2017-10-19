# CNN basic: with MNIST image
'''
1. 이미지 한개를 가져와서
2. CNN Filter로 거르고
3. Max pool Filter로 거른다.
'''

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 1. 이미지 한개를 가져와서
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
# Check out https://www.tensorflow.org/get_started/mnist/beginners for
# more information about the mnist dataset

img = mnist.train.images[0].reshape(28, 28)
plt.imshow(img, cmap='Greys')
plt.show()

# 2. CNN Filter로 거르고
sess = tf.InteractiveSession()

img = img.reshape(-1, 28, 28, 1)
W1 = tf.Variable(tf.random_normal([3, 3, 1, 5], stddev=0.01))   # 3x3 color 1개의 필터 5개
strides = [1, 2, 2, 1]
padding = 'SAME'

conv2d = tf.nn.conv2d(img, W1, strides=strides, padding=padding)
print(conv2d)

sess.run(tf.global_variables_initializer())
conv2d_img = conv2d.eval()
conv2d_img = np.swapaxes(conv2d_img, 0, 3)
for i, one_img in enumerate(conv2d_img):
    plt.subplot(1,5,i+1), plt.imshow(one_img.reshape(14,14), cmap='gray')
plt.show()


# 3. Max pool Filter로 거른다.
ksize = [1, 2, 2, 1]    # 2x2 사이즈 Filter 에서 Max를 찾기
strides = [1, 2, 2, 1]   # 2x2 씩 이동
padding = 'SAME'       # SAME: Input 이미지와 동일한 크기가 나올 수 있도록 패딩을 해줌

pool = tf.nn.max_pool(conv2d, ksize=ksize, strides=strides, padding=padding)
print(pool)

sess.run(tf.global_variables_initializer())
pool_img = pool.eval()
pool_img = np.swapaxes(pool_img, 0, 3)
for i, one_img in enumerate(pool_img):
    plt.subplot(1,5,i+1), plt.imshow(one_img.reshape(7,7), cmap='gray')
plt.show()


'''
Tensor("Conv2D:0", shape=(1, 14, 14, 5), dtype=float32)
Tensor("MaxPool:0", shape=(1, 7, 7, 5), dtype=float32)
'''
