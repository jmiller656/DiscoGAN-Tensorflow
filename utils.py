import tensorflow as tf	
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
def weight_var(shape,name='w'):
	return tf.get_variable(name,shape)

"""
Same as above, but for bias
"""
def bias_var(shape,name='b'):
	return tf.get_variable(name,shape)

"""
Special activation function we'll be using: leaky relu
"""
def lrelu(x,alpha=0.2):
	return tf.maximum(alpha*x,x)

"""
Nice helper function for creating convolutional layers
"""
def conv_layer(x,w_shape,b_shape,activation=tf.nn.relu,batch_norm=True,stride=2,name="conv2d",reuse=False):
	with tf.variable_scope(name) as scope:
		if reuse:
			scope.reuse_variables()
		w = weight_var(w_shape,name="w"+name)
		b = bias_var(b_shape,name="b"+name)
		h = conv2d(x,w,stride=stride)+b
		if batch_norm:
			h = batch_norm_layer(h)
		if activation:
			h = activation(h)
		return h

"""
Nice helper function for creating convolutional transpose layers
"""
def conv_layer_t(x,w_shape,b_shape,shape,activation=tf.nn.relu,batch_norm=True,stride=2,name="deconv2d",reuse=False):
	with tf.variable_scope(name) as scope:
		if reuse:
			scope.reuse_variables()
		w = weight_var(w_shape,name="w"+name)
		b = bias_var(b_shape,name="b"+name)
		h = conv2d_t(x,w,shape)+b
		if batch_norm:
			h = batch_norm_layer(h)
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
		if batch_norm:
			h = batch_norm_layer(h)
		if dropout:
			h = tf.nn.dropout(h,keep_prob)
		return h

"""
Helper function to build Convolutional autoEncoder, used as the generator network
"""
def generator(x,name="generator",reuse=False):
	with tf.variable_scope(name) as scope:
		if reuse:
			scope.reuse_variables()
		conv_1 = conv_layer(x,[4,4,int(x.get_shape()[-1]),64],[64],activation=lrelu,batch_norm=False,name="g_conv_1",reuse=reuse)
		conv_2 = conv_layer(conv_1,[4,4,int(conv_1.get_shape()[-1]),64*2],[64*2],activation=lrelu,name="g_conv_2",reuse=reuse)
		conv_3 = conv_layer(conv_2,[4,4,int(conv_2.get_shape()[-1]),64*4],[64*4],activation=lrelu,name="g_conv_3",reuse=reuse)
		conv_4 = conv_layer(conv_3,[4,4,int(conv_3.get_shape()[-1]),64*8],[64*8],activation=lrelu,name="g_conv_4",reuse=reuse)
		conv_t_1 = conv_layer_t(conv_4,[4,4,64,int(conv_4.get_shape()[-1])],[64],[-1,8,8,64],activation=lrelu,name="g_deconv_1",reuse=reuse)
		conv_t_2 = conv_layer_t(conv_t_1,[4,4,32,int(conv_t_1.get_shape()[-1])],[32],[-1,16,16,32],activation=lrelu,name="g_deconv_2",reuse=reuse)
		conv_t_3 = conv_layer_t(conv_t_2,[4,4,16,int(conv_t_2.get_shape()[-1])],[16],[-1,32,32,16],activation=lrelu,name="g_deconv_3",reuse=reuse)
		conv_t_4 = conv_layer_t(conv_t_3,[4,4,3,int(conv_t_3.get_shape()[-1])],[3],[-1,64,64,3],batch_norm=False,name="g_deconv_4",reuse=reuse)

		out = conv_t_4
	
		return out

"""
Helper function to build discriminator network	
"""
def discriminator(x,name="discriminator",reuse=False):
	with tf.variable_scope(name) as scope:
		if reuse:
			scope.reuse_variables()
		conv_1 = conv_layer(x,[4,4,int(x.get_shape()[-1]),32],[32],activation=lrelu,batch_norm=False,name="d_conv_1",reuse=reuse)
		conv_2 = conv_layer(conv_1,[4,4,int(conv_1.get_shape()[-1]),16],[16],activation=lrelu,name="d_conv_2",reuse=reuse)
		conv_3 = conv_layer(conv_2,[4,4,int(conv_2.get_shape()[-1]),8],[8],activation=lrelu,name="d_conv_3",reuse=reuse)
		conv_4 = conv_layer(conv_3,[4,4,int(conv_3.get_shape()[-1]),4],[4],activation=lrelu,name="d_conv_4",reuse=reuse)
		out = conv_layer(conv_4,[4,4,int(conv_4.get_shape()[-1]),1],[1],activation=None,stride=4,batch_norm=False,name="d_conv_5",reuse=reuse)
		out = tf.squeeze(out)
		return out	
	
