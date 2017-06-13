# TensorFlow 실행
import tensorflow as tf

# Lab 1 - 1
# Python Print로 Version 출력
print(tf.__version__)


# Lab 1 - 2
# 상수의  Default node 선언
hellotensorflow = tf.constant("Hello TensorFlow!")

# Session을 생성
sess = tf.Session()

# Python Print로 출력
print(sess.run(hellotensorflow))
# b'Hello TensorFlow!'


# Lab 1 - 3
node1 = tf.constant(3.0, tf.float32)
node2 = tf.constant(4.0)    # 암시적으로 tf.float32으로 설정된다
node3 = tf.add(node1, node2)

print("node1=", node1, "node2=", node2)
print("node3=", node3)
# node1= Tensor("Const_1:0", shape=(), dtype=float32) node2= Tensor("Const_2:0", shape=(), dtype=float32)
# node3= Tensor("Add:0", shape=(), dtype=float32)

sess = tf.Session()
print("sess.run([node1, node2])=", sess.run([node1, node2]))
print("sess.run(node3)=", sess.run(node3))
# sess.run([node1, node2])= [3.0, 4.0]
# sess.run(node3)= 7.0


# Lab 1 - 4
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder_node = a + b  # tf.add(a, b) 와 같다

print(sess.run(adder_node, feed_dict={a: 3, b: 4.5}))
print(sess.run(adder_node, feed_dict={a: [1, 3], b: [2, 4]}))
# 7.5
# [ 3.  7.]
