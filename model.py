import tensorflow as tf
import data
import utils
import os
import scipy.misc


LR = 2e-4
B1 = 0.5
B2 = 0.999

BATCH_SIZE = 10
model_directory = "models"


#Input placeholders
x_a = tf.placeholder(tf.float32,shape=[BATCH_SIZE,64,64,3],name="xa")
x_b = tf.placeholder(tf.float32,shape=[BATCH_SIZE,64,64,3],name="xb")

#Generator Networks
g_ab = utils.generator(x_a,BATCH_SIZE,name="gen_AB")
g_ba = utils.generator(x_b,BATCH_SIZE,name="gen_BA")

#Secondary generator networks, reusing params of previous two
g_aba = utils.generator(g_ab,BATCH_SIZE,name="gen_BA",reuse=True)
g_bab = utils.generator(g_ba,BATCH_SIZE,name="gen_AB",reuse=True)

#Discriminator for input a
disc_a_real = utils.discriminator(x_a,name="disc_a")
disc_a_fake = utils.discriminator(g_ba,name="disc_a",reuse=True)

#Discriminator for input b
disc_b_real = utils.discriminator(x_b,name="disc_b")
disc_b_fake = utils.discriminator(g_ab,name="disc_b",reuse=True)

#Reconstruction loss for generators
l_const_a = tf.reduce_mean(utils.huber_loss(g_aba,x_a))
l_const_b = tf.reduce_mean(utils.huber_loss(g_bab,x_b))

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

#Parameter lists
disc_params = []
gen_params = []

for v in tf.trainable_variables():
	if 'disc' in v.name:
		disc_params.append(v)
	if 'gen' in v.name:
		gen_params.append(v)


trainer_Da = tf.train.AdamOptimizer(LR,beta1=B1,beta2=B2).minimize(l_disc,var_list=disc_params)
trainer_G = tf.train.AdamOptimizer(LR,beta1=B1,beta2=B2).minimize(l_g,var_list=gen_params)

init = tf.global_variables_initializer()	

if not os.path.exists('generator a->b'):
		os.makedirs('generator a->b')
if not os.path.exists('generator b->a'):
		os.makedirs('generator b->a')
if not os.path.exists('reconstruct b'):
		os.makedirs('reconstruct b')
if not os.path.exists('reconstruct a'):
		os.makedirs('reconstruct a')

with tf.Session() as sess:
	
	sess.run(init)
	saver = tf.train.Saver()

	try:
		saver.restore(sess, tf.train.latest_checkpoint("models"))
	except:
		print("Previous weights not found")		

	for i in range(50000):

		realA = data.get_batch_a(BATCH_SIZE)
		realB = data.get_batch_b(BATCH_SIZE)

		_,dLoss = sess.run([trainer_D,l_disc],feed_dict={x_a:realA,x_b:realB})	
	
		_,gLoss = sess.run([trainer_G,l_g],feed_dict={x_a:realA,x_b:realB})
		
		realA = data.get_batch_a(BATCH_SIZE)
		realB = data.get_batch_b(BATCH_SIZE)

		_,gLoss = sess.run([trainer_G,l_g],feed_dict={x_a:realA,x_b:realB})

		print("Generator Loss: " + str(gLoss) + "\tDiscriminator Loss: " + str(dLoss)) 
		
		if i %10 == 0:
			out_a,out_b,out_ab,out_ba = sess.run([g_ba,g_ab,g_aba,g_bab],feed_dict={x_a:realA,x_b:realB})
			scipy.misc.imsave("a/gen"+str(i%500)+'.png',data.postprocess(out_a[0]))
			scipy.misc.imsave("b/gen"+str(i%500)+'.png',data.postprocess(out_b[0]))
			scipy.misc.imsave("ba/gen"+str(i%500)+'.png',data.postprocess(out_ba[0]))
			scipy.misc.imsave("ab/gen"+str(i%500)+'.png',data.postprocess(out_ab[0]))
		if i % 1000 == 0:
			if not os.path.exists(model_directory):
				os.makedirs(model_directory)
			saver.save(sess,model_directory+'/model-'+str(i)+'.ckpt')
			print("Saved Model")



