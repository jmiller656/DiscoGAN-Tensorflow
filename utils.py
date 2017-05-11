import tensorflow as tf	

class batch_norm(object):
  def __init__(self, epsilon=1e-3, momentum = 0.9, name="batch_norm"):
    with tf.variable_scope(name):
      self.epsilon  = epsilon
      self.momentum = momentum
      self.name = name

  def __call__(self, x, train=True):
    return tf.contrib.layers.batch_norm(x,
                      decay=self.momentum, 
                      updates_collections=None,
                      epsilon=self.epsilon,
                      scale=True,
                      is_training=train,
                      scope=self.name)

"""
Helper for convolution function given an input and conv weights
"""
def conv2d(x,w,stride=2,padding="SAME"):
	return tf.nn.conv2d(x,w,strides=[1,stride,stride,1],padding=padding)

"""
Helper for convolution transpose function given an input and conv weights
"""
def conv2d_t(x,w,shape,stride=2,padding="SAME"):
	return tf.nn.conv2d_transpose(x,w,shape,strides=[1,stride,stride,1],padding=padding)


"""
Helper for weight variable given shape. Does fancy initialization
"""
def weight_var(shape,name='w',init=tf.truncated_normal_initializer(stddev=0.02)):
	return tf.get_variable(name,shape,initializer=init)

"""
Same as above, but for bias
"""
def bias_var(shape,name='b'):
	return tf.get_variable(name,shape,initializer=tf.constant_initializer(0.0))

"""
Special activation function we'll be using: leaky relu
"""
def lrelu(x,alpha=0.2):
	return tf.maximum(alpha*x,x)

"""
Nice helper function for creating convolutional layers
"""
def conv_layer(x,w_shape,b_shape,activation=tf.nn.relu,batch_norm=None,stride=2,name="conv2d",reuse=False):
	with tf.variable_scope(name) as scope:
		if reuse:
			scope.reuse_variables()
		w = weight_var(w_shape,name="w"+name)
		b = bias_var(b_shape,name="b"+name)
		h = conv2d(x,w,stride=stride)+b
		if batch_norm is not None:
			h = batch_norm(h)
		if activation:
			h = activation(h)
		return h

"""
Nice helper function for creating convolutional transpose layers
"""
def conv_layer_t(x,w_shape,b_shape,shape,activation=tf.nn.relu,batch_norm=None,stride=2,name="deconv2d",reuse=False):
	with tf.variable_scope(name) as scope:
		if reuse:
			scope.reuse_variables()
		w = weight_var(w_shape,name="w"+name)
		b = bias_var(b_shape,name="b"+name)
		h = conv2d_t(x,w,shape)+b
		if batch_norm is not None:
			h = batch_norm(h)
		if activation:
			h = activation(h)
		return h

"""
Another helper for batch norm. This should help a bit
"""
def batch_norm_layer(x,is_training=True):
	return tf.contrib.layers.batch_norm(x,is_training=is_training,epsilon=1e-4,trainable=True)

"""
Helper function for creating fully connected layers. Droupout and batchnorm optional
"""
def fc_layer(x,w_shape,b_shape,activation=tf.nn.relu,batch_norm=True,dropout=True,name="linear",reuse=False):
	with tf.variable_scope(name) as scope:
		if reuse:
			scope.reuse_variables()
		w = weight_var(w_shape)
		b = bias_var(b_shape)
		h = tf.matmul(x,w)+b
		if activation:
			h = activation(h)
		if batch_norm is not None:
			h = batch_norm(h)
		if dropout:
			h = tf.nn.dropout(h,keep_prob)
		return h

"""
Huber loss function
"""
def huber_loss(logits,labels,max_gradient=1.0):
	err = tf.abs(labels-logits)
	mg = tf.constant(max_gradient)
	lin = mg*(err-0.5*mg)
	quad = 0.5*err*err
	return tf.where(err<mg,quad,lin)

"""
Helper function to build Convolutional autoEncoder, used as the generator network
"""
def generator(x,name="generator",im_size=64,channels=3,reuse=False):
	with tf.variable_scope(name) as scope:
		if reuse:
			scope.reuse_variables()

		g_bn0 = batch_norm(name='g_bn0')
		g_bn1 = batch_norm(name='g_bn1')
		g_bn2 = batch_norm(name='g_bn2')
		g_bn3 = batch_norm(name='g_bn3')
		g_bn4 = batch_norm(name='g_bn4')
		g_bn5 = batch_norm(name='g_bn5')

		conv_1 = conv_layer(x,[4,4,int(x.get_shape()[-1]),im_size],[im_size],activation=lrelu,batch_norm=None,name="g_conv_1",reuse=reuse)
		conv_2 = conv_layer(conv_1,[4,4,int(conv_1.get_shape()[-1]),im_size*2],[im_size*2],activation=lrelu,batch_norm=g_bn0,name="g_conv_2",reuse=reuse)
		conv_3 = conv_layer(conv_2,[4,4,int(conv_2.get_shape()[-1]),im_size*4],[im_size*4],activation=lrelu,batch_norm=g_bn1,name="g_conv_3",reuse=reuse)
		conv_4 = conv_layer(conv_3,[4,4,int(conv_3.get_shape()[-1]),im_size*8],[im_size*8],activation=lrelu,batch_norm=g_bn2,name="g_conv_4",reuse=reuse)
		conv_t_1 = conv_layer_t(conv_4,[4,4,im_size,int(conv_4.get_shape()[-1])],[im_size],[tf.shape(x)[0],im_size/8,im_size/8,im_size],activation=lrelu,batch_norm=g_bn3,name="g_deconv_1",reuse=reuse)
		conv_t_2 = conv_layer_t(conv_t_1,[4,4,im_size/2,int(conv_t_1.get_shape()[-1])],[im_size/2],[tf.shape(x)[0],im_size/4,im_size/4,im_size/2],activation=lrelu,batch_norm=g_bn4,name="g_deconv_2",reuse=reuse)
		conv_t_3 = conv_layer_t(conv_t_2,[4,4,im_size/4,int(conv_t_2.get_shape()[-1])],[im_size/4],[tf.shape(x)[0],im_size/2,im_size/2,im_size/4],activation=lrelu,batch_norm=g_bn5,name="g_deconv_3",reuse=reuse)
		conv_t_4 = conv_layer_t(conv_t_3,[4,4,channels,int(conv_t_3.get_shape()[-1])],[channels],[tf.shape(x)[0],im_size,im_size,3],activation=None,batch_norm=None,name="g_deconv_4",reuse=reuse)

		out = conv_t_4
	
		return out

"""
Helper function to build discriminator network	
"""
def discriminator(x,name="discriminator",im_size=64,reuse=False):
	with tf.variable_scope(name) as scope:
		if reuse:
			scope.reuse_variables()

		d_bn0 = batch_norm(name='d_bn0')
		d_bn1 = batch_norm(name='d_bn1')
		d_bn2 = batch_norm(name='d_bn2')		

		conv_1 = conv_layer(x,[4,4,int(x.get_shape()[-1]),im_size/2],[im_size/2],activation=lrelu,batch_norm=None,name="d_conv_1",reuse=reuse)
		conv_2 = conv_layer(conv_1,[4,4,int(conv_1.get_shape()[-1]),im_size/4],[im_size/4],activation=lrelu,batch_norm=d_bn0,name="d_conv_2",reuse=reuse)
		conv_3 = conv_layer(conv_2,[4,4,int(conv_2.get_shape()[-1]),im_size/8],[im_size/8],activation=lrelu,batch_norm=d_bn1,name="d_conv_3",reuse=reuse)
		conv_4 = conv_layer(conv_3,[4,4,int(conv_3.get_shape()[-1]),im_size/16],[im_size/16],activation=lrelu,batch_norm=d_bn2,name="d_conv_4",reuse=reuse)
		out = conv_layer(conv_4,[4,4,int(conv_4.get_shape()[-1]),1],[1],activation=None,stride=4,batch_norm=None,name="d_conv_5",reuse=reuse)
		out = tf.squeeze(out)
		return out
	

