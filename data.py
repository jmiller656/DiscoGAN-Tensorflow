import numpy as np
import cv2
import os
import random
import scipy.misc

"""
Get batch from domain
"""
def get_batch(size,domain):
	samples = os.listdir("images/"+domain)
	images =[]
	indices = random.sample(range(0,len(samples)),size)
	for i in indices:
		img = scipy.misc.imread("images/"+domain+"/"+samples[i])
		img = preprocess(img)
		images.append(img)
	return np.asarray(images)

"""
Preprocess image
"""
def preprocess(img):
	img = scipy.misc.imresize(img,[64,64])
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
			

