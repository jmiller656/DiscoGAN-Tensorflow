import tensorflow as tf
import utils

a = tf.placeholder(tf.float32,[1,64,64,3])
g = utils.generator(a)
d = utils.discriminator(g)
print d
