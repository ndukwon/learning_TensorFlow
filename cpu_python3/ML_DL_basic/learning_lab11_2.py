# CNN basic: max pool filter

import numpy as np
import tensorflow as tf

sess = tf.InteractiveSession()
image = np.array([[[[4],[3]],
                   [[2],[1]]]], dtype=np.float32)

ksize = [1, 2, 2, 1]    # 2x2 사이즈 Filter 에서 Max를 찾기
stride = [1, 1, 1, 1]   # 1x1 씩 이동
# padding = 'VALID'       # VALID: 패딩을 하지 않음
padding = 'SAME'       # SAME: Input 이미지와 동일한 크기가 나올 수 있도록 패딩을 해줌

pool = tf.nn.max_pool(image, ksize=ksize, strides=stride, padding=padding)

print("pool.shape:", pool.shape)
print("pool.eval()", pool.eval())


'''
# padding = 'VALID'
pool.shape: (1, 1, 1, 1)
pool.eval() [[[[ 4.]]]]

# padding = 'SAME'
pool.shape: (1, 2, 2, 1)
pool.eval() [[[[ 4.]
   [ 3.]]

  [[ 2.]
   [ 1.]]]]
'''
