# Tensor manipulate

import tensorflow as tf
import numpy as np
import pprint
pp = pprint.PrettyPrinter(indent=4)

# Simple Array
print("# Simple Array")
t = np.array([0., 1., 2., 3., 4., 5., 6.])
pp.pprint(t)                        # array([ 0.,  1.,  2.,  3.,  4.,  5.,  6.])

print(t.ndim)                       # 1                     => Rank: Matrix Dimention을 표현
print(t.shape)                      # 1 (7,)                => Shape: 1 x 7 Matrix와 같은 사이즈를 표현
print(t[0], t[1], t[-1], t[-2])     # 0.0 1.0 6.0 5.0       => -가 들어가면 length에서 뺀 index(0부터 시작)
print(t[2:5], t[4:-1])              # [ 2.  3.  4.] [ 4.  5.] => 시작 index는 포함되고, 끝 index는 포함되지 않는다.
print(t[:2], t[3:])                 # [ 0.  1.] [ 3.  4.  5.  6.]

# 2D Array
print("# 2D Array")
t = np.array([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.], [10., 11., 12.]])
pp.pprint(t)
'''
array([[  1.,   2.,   3.],
       [  4.,   5.,   6.],
       [  7.,   8.,   9.],
       [ 10.,  11.,  12.]])
'''

print(t.ndim)                       # 2                     => Rank: Matrix Dimention을 표현
print(t.shape)                      # (4, 3)                => Shape: 4 x 3 Matrix 사이즈를 표현


# Open session
sess = tf.InteractiveSession()

# Tensorflow shape
print("# Tensorflow shape")
t = tf.constant([1, 2, 3, 4])
pp.pprint(t.eval())                 # array([1, 2, 3, 4], dtype=int32)
pp.pprint(tf.shape(t).eval())       # array([4], dtype=int32)

t = tf.constant([[1, 2], [3, 4]])
pp.pprint(t.eval())
'''
array([[1, 2],
       [3, 4]], dtype=int32)
'''
pp.pprint(tf.shape(t).eval())       # array([2, 2], dtype=int32)

t = tf.constant([[[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]],[[13, 14, 15, 16], [17, 18, 19, 20], [21, 22, 23, 24]]]])
pp.pprint(t.eval())
'''
array([[[[ 1,  2,  3,  4],
         [ 5,  6,  7,  8],
         [ 9, 10, 11, 12]],

        [[13, 14, 15, 16],
         [17, 18, 19, 20],
         [21, 22, 23, 24]]]], dtype=int32)
'''
pp.pprint(tf.shape(t).eval())       # array([1, 2, 3, 4], dtype=int32)


# Matmul VS multiply
print("# Matmul VS multiply")
matrix1 = tf.constant([3., 3.])
pp.pprint(tf.shape(matrix1).eval())     # array([2], dtype=int32)
matrix1 = tf.constant([[3., 3.]])
pp.pprint(tf.shape(matrix1).eval())     # array([1, 2], dtype=int32)

matrix2 = tf.constant([[2.],[2.]])
pp.pprint(matrix2.eval())
'''
matrix1 = [3 3] (1 * 2 Matrix)
matrix2 = [2    (2 * 1 Matrix)
           2]
'''
pp.pprint(tf.matmul(matrix1, matrix2).eval())
pp.pprint((matrix1 * matrix2).eval())
'''
array([[ 12.]], dtype=float32)
array([[ 6.,  6.],
       [ 6.,  6.]], dtype=float32)
'''

pp.pprint((matrix1 + matrix2).eval())
'''
array([[ 5.,  5.],
       [ 5.,  5.]], dtype=float32)
'''

matrix1 = tf.constant([[3., 3.]])
matrix2 = tf.constant([[2., 2.]])
pp.pprint((matrix1+matrix2).eval())
'''
array([[ 5.,  5.]], dtype=float32)
'''

# Reduce mean
print("# Reduce mean")
print(tf.reduce_mean([1, 2]).eval())        # 1     => Element가 int기 때문에 1.5가 아닌 1이 나옴

x = [[1., 2.], [3., 4.]]
print(tf.reduce_mean(x).eval())             # 2.5
print(tf.reduce_mean(x, axis=0).eval())     # [ 2.  3.]
print(tf.reduce_mean(x, axis=1).eval())     # [ 1.5  3.5]
print(tf.reduce_mean(x, axis=-1).eval())    # [ 1.5  3.5]   => 마지막 Dimention(안쪽으로 들어가는 방향) 즉, 1

# Reduce sum
print("# Reduce sum")
print(tf.reduce_sum(x).eval())              # 10.0
print(tf.reduce_sum(x, axis=0).eval())      # [ 4.  6.]
print(tf.reduce_sum(x, axis=1).eval())      # [ 3.  7.]
print(tf.reduce_sum(x, axis=-1).eval())     # [ 3.  7.]

# Argmax
print("# Argmax")
x = [[0, 1, 2], [2, 1, 0]]
print(tf.argmax(x, axis=0).eval())          # [1 0 0]   => index로 나옴
print(tf.argmax(x, axis=1).eval())          # [2 0]     => 세로지만 Array로 나옴
print(tf.argmax(x, axis=-1).eval())         # [2 0]


# Reshape
print("# Reshape")
t = np.array([[[0, 1, 2],
               [3, 4, 5]],

              [[6, 7, 8],
               [9, 10, 11]]])
pp.pprint(t)
print(t.shape)
'''
array([[[ 0,  1,  2],
        [ 3,  4,  5]],

       [[ 6,  7,  8],
        [ 9, 10, 11]]])
(2, 2, 3)
'''

pp.pprint(tf.reshape(t, shape=[-1, 3]).eval())      # 가장 안쪽에 3개씩
'''
array([[ 0,  1,  2],
       [ 3,  4,  5],
       [ 6,  7,  8],
       [ 9, 10, 11]])
'''
pp.pprint(tf.reshape(t, shape=[-1, 2]).eval())      # 가장 안쪽에 2개씩
'''
array([[ 0,  1],
       [ 2,  3],
       [ 4,  5],
       [ 6,  7],
       [ 8,  9],
       [10, 11]])
'''

pp.pprint(tf.reshape(t, shape=[-1, 1, 3]).eval())   # 가장 안쪽에 3개씩하고 1차원 추가
'''
array([[[ 0,  1,  2]],

       [[ 3,  4,  5]],

       [[ 6,  7,  8]],

       [[ 9, 10, 11]]])
'''

# pp.pprint(tf.reshape(t, shape=[-1, 2, 4]).eval())   # error
pp.pprint(tf.reshape(t, shape=[-1, 3, 4]).eval())   # 가장 안쪽에 4개씩하고 ??
'''
array([[[ 0,  1,  2,  3],
        [ 4,  5,  6,  7],
        [ 8,  9, 10, 11]]])
'''

print(tf.squeeze([[0], [1], [2]]).eval())           # 하나짜리는 밖으로 꺼낸다.
print(tf.squeeze([[0, 1], [1, 2], [2, 3]]).eval())  # 하나짜리가 아니라서 나오지 않는다.
'''
[0 1 2]
[[0 1]
 [1 2]
 [2 3]]
'''

print(tf.expand_dims([0, 1, 2], 1).eval())
# print(tf.expand_dims([0, 1, 2], 2).eval())        # error
# print(tf.expand_dims([[0], [1], [2]]).eval())     # error
# print(tf.expand_dims([0, 1, 2]).eval())           # error
'''
[[0]
 [1]
 [2]]
'''

# one_hot
# 주어진 Index를 1을 표시하는 Matrix로 만드는것
# 주의: depth가 하나 더 추가 되기 때문에 reshape를 해준다.
print("tf.one_hot([[0], [1], [2], [0]], depth=3).eval()")
R = tf.one_hot([[0], [1], [2], [0]], depth=3).eval()
print(R)
print(tf.reshape(R, shape=[-1, 3]).eval())
'''
[[[ 1.  0.  0.]]

 [[ 0.  1.  0.]]

 [[ 0.  0.  1.]]

 [[ 1.  0.  0.]]]

[[ 1.  0.  0.]
 [ 0.  1.  0.]
 [ 0.  0.  1.]
 [ 1.  0.  0.]]
'''

# Cast
# Type casting을 한다.
# 소숫점 -> int -> 소숫점 이하 절사
# boolean -> int -> True:1, False:0
# int -> boolean -> 0:False, 0이 아닌것:True
print("tf.cast([1.8, 2.2, 3.3, 4.9], tf.int32).eval()")
print(tf.cast([1.8, 2.2, 3.3, 4.9], tf.int32).eval())
print("tf.cast([True, False, 1 == 1, 1 == 0], tf.int32).eval()")
print([True, False, 1 == 1, 1 == 0])
print(tf.cast([True, False, 1 == 1, 1 == 0], tf.int32).eval())
print(tf.cast([1, 99, 0], tf.bool).eval())
'''
[1 2 3 4]

[True, False, True, False]
[1 0 1 0]
[ True  True False]
'''


# Stack
# 배열을 위쪽, 왼쪽부터 쌓아서 Matrix를 만든다.
print("\n# Stack")

x = [1, 2]
y = [3, 4]
z = [5, 6]

new = [x, y, z]
print(new)
if __name__ == '__main__':
    print(tf.stack(new).eval())             # 세로로, 아래로 붙는다.
print(tf.stack(new, axis=1).eval())         # 가로로, 오른쪽에 붙는다.
print(tf.stack(new, axis=-1).eval())        # 가로로, 오른쪽에 붙는다.
'''
[[1, 2], [3, 4], [5, 6]]
[[1 2]
 [3 4]
 [5 6]]
[[1 3 5]
 [2 4 6]]
[[1 3 5]
 [2 4 6]]
'''


# Ones, Zeros like
# 주어진 배열과 똑같은 사이즈에 Ones:1, Zeros:0 으로 채워진 Matrix를 만든다
print("\n# Ones, Zeros like")
x = [[0, 1, 2], [2, 1, 0]]
print(tf.ones_like(x).eval())
print(tf.zeros_like(x).eval())
'''
[[1 1 1]
 [1 1 1]]
[[0 0 0]
 [0 0 0]]
'''


# Zip
# 변수에 각각의 스트리밍을 받아서 돌 수 있게 한다.
print("\n# Zip")

for x, y in zip([1, 2, 3], [4, 5, 6]):
    print(x, y)
# for x, y in zip([[1, 2, 3], [4, 5, 6]]):
#     print(x, y)
'''
1 4
2 5
3 6
'''
