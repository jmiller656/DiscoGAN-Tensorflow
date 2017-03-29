import numpy as np
import cv2
import os
import random
import scipy.misc

marios = os.listdir("sprite/mario")
sonics = os.listdir("sprite/sonic")

def get_batch_a(size):
	images =[]
	indices = random.sample(range(0,len(marios)),size)
	for i in indices:
		img = scipy.misc.imread("sprite/mario/"+marios[i])
		img = scipy.misc.imresize(img,[64,64])
		img = img.astype(np.float32)/127.5 - 1.
		images.append(img)
	return np.asarray(images).astype(np.float32)

def get_batch_b(size):
	images =[]
        indices = random.sample(range(0,len(sonics)),size)
        for i in indices:
                img = scipy.misc.imread("sprite/sonic/"+sonics[i])
		img = scipy.misc.imresize(img,[64,64])
		img = img.astype(np.float32)/127.5 - 1.
		images.append(img)
	return np.asarray(images).astype(np.float32)
