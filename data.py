import numpy as np
import zipfile
import cv2
import os
import random
import scipy.misc
from StringIO import StringIO

domainA = []
domainB = []

im_size = 64

"""
Preload celeba dataset
"""
def load_celeba():
	print "loading celeba"
	f = open('images/list_attr_celeba.txt')
	t = f.readline()
	t = f.readline()
	t = f.readline()
	while t:
		strs = t.split()
		fname = strs[0]
		att = int(strs[3])
		if att == -1:
			domainA.append(fname)
		else:
			domainB.append(fname)
		t = f.readline()


def set_im_size(size):
	im_size = size

"""
Get batch from domain
"""
def get_batch(size,domain):
	if len(domainA) == 0 or len(domainB) == 0:
		load_celeba()
	"""
	samples = os.listdir("images/"+domain)
	images =[]
	indices = random.sample(range(0,len(samples)),size)
	for i in indices:
		img = scipy.misc.imread("images/"+domain+"/"+samples[i])
		img = preprocess(img)
		images.append(img)
	return np.asarray(images)"""
	if domain == 'a':
		samples = domainA
	else:
		samples = domainB
	images = []
	indices = random.sample(range(0,len(samples)),size)
	zfile = zipfile.ZipFile('images/celeba.zip','r')
	for i in indices:
		data = zfile.read('img_align_celeba/'+samples[i])
		img = scipy.misc.imread(StringIO(data)) 
		img = preprocess(img)
		images.append(img)

	return np.asarray(images)
	
	

"""
Preprocess image
"""
def preprocess(img):
	img = scipy.misc.imresize(img,[im_size,im_size])
	img = img.astype(np.float32)/127.5 - 1.
	return img

"""
Postprocess image
"""
def postprocess(img):
	return (img+1.)/2

"""
Save image
"""
def save(filename,img):
	scipy.misc.imsave(filename,postprocess(img))
			

