# CNN basic: filter

'''
3x3 이미지를 2x2 필터로 추출
'''

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


sess = tf.InteractiveSession()
image = np.array([[[[1], [2], [3]],
                   [[4], [5], [6]],
                   [[7], [8], [9]]]], dtype=np.float32)

print("image.shape:", image.shape)
# image.shape: (1, 3, 3, 1) => 이미지 개수, 사이즈 세로, 사이즈 가로, 컬러

plt.imshow(image.reshape(3, 3), cmap='Greys')
plt.show()

# Filter 1개
# weight = tf.constant([[[[1.]],[[1.]]],
#                       [[[1.]],[[1.]]]])

# Filter 3개
weight = tf.constant([[[[1.,10.,-1.]],[[1.,10.,-1.]]],
                      [[[1.,10.,-1.]],[[1.,10.,-1.]]]])

print("weight.shape:", weight.shape)
# weight.shape: (2, 2, 1, 1) => 필터 세로, 필터 가로, 컬러, 필터 개수

# stride: 1x1(한칸씩 움직인다)
stride = [1, 1, 1, 1]
# padding = 'VALID'
padding = 'SAME'

conv2d = tf.nn.conv2d(image, weight, strides=stride, padding=padding)
conv2d_img = conv2d.eval()
print("conv2d_img.shape:", conv2d_img.shape)
# conv2d_img.shape: (1, 2, 2, 1)

# plt.imshow(conv2d_img.reshape(2, 2), cmap='Greys')
# plt.imshow(conv2d_img.reshape(3, 3), cmap='Greys')
# plt.show()

conv2d_img = np.swapaxes(conv2d_img, 0, 3)
for i, one_img in enumerate(conv2d_img):
    # print(one_img.reshape(2,2))
    # plt.subplot(1,2,i+1), plt.imshow(one_img.reshape(2,2), cmap='gray')
    print(one_img.reshape(3,3))
    plt.subplot(1, 3, i+1), plt.imshow(one_img.reshape(3,3), cmap='gray')
plt.show()
