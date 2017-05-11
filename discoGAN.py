import tensorflow as tf
import utils
import data
import os

"""
This object represents a discogan
machine learning model. It comes with
functions to train and restore weights
for a DiscoGAN
"""
class DiscoGAN(object):
	def __init__(self,batch_size=10,im_size=64,channels=3,dtype=tf.float32,analytics=True):
		self.analytics = analytics
		self.batch_size = batch_size

		self.x_a = tf.placeholder(dtype,[None,im_size,im_size,channels],name='xa')
		self.x_b = tf.placeholder(dtype,[None,im_size,im_size,channels],name='xb')

		#Generator Networks
		self.g_ab = utils.generator(self.x_a,name="gen_AB",im_size=im_size)
		self.g_ba = utils.generator(self.x_b,name="gen_BA",im_size=im_size)

		#Secondary generator networks, reusing params of previous two
		self.g_aba = utils.generator(self.g_ab,name="gen_BA",im_size=im_size,reuse=True)
		self.g_bab = utils.generator(self.g_ba,name="gen_AB",im_size=im_size,reuse=True)

		#Discriminator for input a
		self.disc_a_real = utils.discriminator(self.x_a,name="disc_a",im_size=im_size)
		self.disc_a_fake = utils.discriminator(self.g_ba,name="disc_a",im_size=im_size,reuse=True)

		#Discriminator for input b
		self.disc_b_real = utils.discriminator(self.x_b,name="disc_b")
		self.disc_b_fake = utils.discriminator(self.g_ab,name="disc_b",reuse=True)

		#Reconstruction loss for generators
		self.l_const_a = tf.reduce_mean(utils.huber_loss(self.g_aba,self.x_a))
		self.l_const_b = tf.reduce_mean(utils.huber_loss(self.g_bab,self.x_b))

		#Generation loss for generators 
		self.l_gan_a = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.disc_a_fake,labels=tf.ones_like(self.disc_a_fake)))
		self.l_gan_b = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.disc_b_fake,labels=tf.ones_like(self.disc_b_fake)))

		#Real example loss for discriminators
		self.l_disc_a_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.disc_a_real,labels=tf.ones_like(self.disc_a_real)))
		self.l_disc_b_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.disc_b_real,labels=tf.ones_like(self.disc_b_real)))

		#Fake example loss for discriminators
		self.l_disc_a_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.disc_a_fake,labels=tf.zeros_like(self.disc_a_fake)))
		self.l_disc_b_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.disc_b_fake,labels=tf.zeros_like(self.disc_b_fake)))

		#Combined loss for individual discriminators
		self.l_disc_a = self.l_disc_a_real + self.l_disc_a_fake
		self.l_disc_b = self.l_disc_b_real + self.l_disc_b_fake

		#Total discriminator loss
		self.l_disc = self.l_disc_a + self.l_disc_b

		#Combined loss for individual generators
		self.l_ga = self.l_gan_a + self.l_const_b
		self.l_gb = self.l_gan_b + self.l_const_a

		#Total GAN loss
		self.l_g = self.l_ga + self.l_gb

		#Parameter Lists
		self.disc_params = []
		self.gen_params = []

		for v in tf.trainable_variables():
			if 'disc' in v.name:
				self.disc_params.append(v)
			if 'gen' in v.name:
				self.gen_params.append(v)

		if self.analytics:
			self.init_analytics()

		self.gen_a_dir = 'generator a->b'
		self.gen_b_dir = 'generator b->a'
		self.rec_a_dir = 'reconstruct a'
		self.rec_b_dir = 'reconstruct b'
		self.model_directory = "models"	
	
		if not os.path.exists(self.gen_a_dir):
			os.makedirs(self.gen_a_dir)
		if not os.path.exists(self.gen_b_dir):
			os.makedirs(self.gen_b_dir)
		if not os.path.exists(self.rec_b_dir):
			os.makedirs(self.rec_b_dir)
		if not os.path.exists(self.rec_a_dir):
			os.makedirs(self.rec_a_dir)	

		self.sess = tf.Session()
		self.saver = tf.train.Saver()

	"""
	Enable logging of analytics
	for tensorboard
	"""
	def init_analytics(self):
		#Scalars for all losses
		tf.summary.scalar("loss_g", self.l_g)
		tf.summary.scalar("loss_ga", self.l_ga)
		tf.summary.scalar("loss_gb", self.l_gb)
		tf.summary.scalar("loss_d", self.l_disc)
		tf.summary.scalar("loss_d_a", self.l_disc_a)
		tf.summary.scalar("loss_d_b", self.l_disc_b)
		tf.summary.scalar("l_const_a",self.l_const_a)
		tf.summary.scalar("l_const_b",self.l_const_b)
		
		#Histograms for all vars
		for v in tf.trainable_variables():
			tf.summary.histogram(v.name,v)
		
		self.merged_summary_op = tf.summary.merge_all()
		
	"""
	Train DiscoGAN
	"""
	def train(self,LR=2e-4,B1=0.5,B2=0.999,iterations=50000,sample_frequency=10,
	sample_overlap=500,save_frequency=1000,domain_a="a",domain_b="b"):
		self.trainer_D = tf.train.AdamOptimizer(LR,beta1=B1,beta2=B2).minimize(self.l_disc,var_list=self.disc_params)
		self.trainer_G = tf.train.AdamOptimizer(LR,beta1=B1,beta2=B2).minimize(self.l_g,var_list=self.gen_params)

		with self.sess as sess:
			sess.run(tf.global_variables_initializer())
			if self.analytics:
				if not os.path.exists("logs"):
					os.makedirs("logs")	
				self.summary_writer = tf.summary.FileWriter(os.getcwd()+'/logs',graph=sess.graph)
			for i in range(iterations):
				realA = data.get_batch(self.batch_size,domain_a)
				realB = data.get_batch(self.batch_size,domain_b)
				op_list = [self.trainer_D,self.l_disc,self.trainer_G,self.l_g,self.merged_summary_op]

				_,dLoss,_,gLoss,summary_str = sess.run(op_list,feed_dict={self.x_a:realA,self.x_b:realB})	
			
				realA = data.get_batch(self.batch_size,domain_a)
				realB = data.get_batch(self.batch_size,domain_b)

				_,gLoss = sess.run([self.trainer_G,self.l_g],feed_dict={self.x_a:realA,self.x_b:realB})

				if i%10 == 0:
					self.summary_writer.add_summary(summary_str, i)

				print("Generator Loss: " + str(gLoss) + "\tDiscriminator Loss: " + str(dLoss)) 
			
				if i % sample_frequency == 0:
					realA = data.get_batch(1,domain_a)
					realB = data.get_batch(1,domain_b)
					ops = [self.g_ba,self.g_ab,self.g_aba,self.g_bab]
					out_a,out_b,out_ab,out_ba = sess.run(ops,feed_dict={self.x_a:realA,self.x_b:realB})
					data.save(self.gen_a_dir+"/img"+str(i%sample_overlap)+'.png',out_a[0])
					data.save(self.gen_b_dir+"/img"+str(i%sample_overlap)+'.png',out_b[0])
					data.save(self.rec_a_dir+"/img"+str(i%sample_overlap)+'.png',out_ba[0])
					data.save(self.rec_b_dir+"/img"+str(i%sample_overlap)+'.png',out_ab[0])
				if i % save_frequency == 0:
					if not os.path.exists(self.model_directory):
						os.makedirs(self.model_directory)
					self.saver.save(sess,self.model_directory+'/model-'+str(i)+'.ckpt')
					print("Saved Model")

		"""
		Restore previously saved weights from
		trained / in-progress model
		"""
		def restore():
			try:
				self.saver.restore(self.sess, tf.train.latest_checkpoint(self.model_directory))
			except:
				print("Previous weights not found")


		
