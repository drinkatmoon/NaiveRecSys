print(11)


import tensorflow as tf
tf.__version__
hello = tf.constant('hello,tf')
sess = tf.Session()
print(sess.run(hello))