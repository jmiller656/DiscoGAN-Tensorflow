import tensorflow as tf
import utils

LR = 2e-4
B1 = 0.5
B2 = 0.999

#Input placeholders
x_a = tf.placeholder(tf.float32,shape=[200,64,64,3],name="xa")
x_b = tf.placeholder(tf.float32,shape=[200,64,64,3],name="xb")

#Generator Networks
g_ab = utils.generator(x_a,name="gen_AB")
g_ba = utils.generator(x_b,name="gen_BA")

#Secondary generator networks, reusing params of previous two
g_aba = utils.generator(g_ab,name="gen_BA",reuse=True)
g_bab = utils.generator(g_ba,name="gen_AB",reuse=True)

#Discriminator for input a
disc_a_real = utils.discriminator(x_a,name="disc_a")
disc_a_fake = utils.discriminator(g_ba,name="disc_a",reuse=True)

#Discriminator for input b
disc_b_real = utils.discriminator(x_b,name="disc_b")
disc_b_fake = utils.discriminator(g_ab,name="disc_b",reuse=True)

#Reconstruction loss for generators
l_const_a = tf.nn.l2_loss(g_aba-x_a)
l_const_b = tf.nn.l2_loss(g_bab-x_b)

#Generation loss for generators 
l_gan_a = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_a_fake,labels=tf.ones_like(disc_a_fake)))
l_gan_b = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_b_fake,labels=tf.ones_like(disc_b_fake)))

#Real example loss for discriminators
l_disc_a_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_a_real,labels=tf.ones_like(disc_a_real)))
l_disc_b_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_b_real,labels=tf.ones_like(disc_b_real)))

#Fake example loss for discriminators
l_disc_a_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_a_fake,labels=tf.zeros_like(disc_a_fake)))
l_disc_b_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_b_fake,labels=tf.zeros_like(disc_b_fake)))

#Combined loss for individual discriminators
l_disc_a = l_disc_a_real + l_disc_a_fake
l_disc_b = l_disc_b_real + l_disc_b_fake

#Total discriminator loss
l_disc = l_disc_a + l_disc_b

#Combined loss for individual generators
l_ga = l_gan_a + l_const_a
l_gb = l_gan_b + l_const_b

#Total GAN loss
l_g = l_ga + l_gb

disc_params = []
gen_params = []

for v in tf.trainable_variables():
	if 'disc' in v.name:
		disc_params.append(v)
	if 'gen' in v.name:
		gen_params.append(v)
for q in disc_params:
	print q

for q in gen_params:
	print q

trainer_D = tf.train.AdamOptimizer(LR,beta1=B1,beta2=B2)
trainer_G = tf.train.AdamOptimizer(LR,beta1=B1,beta2=B2)


