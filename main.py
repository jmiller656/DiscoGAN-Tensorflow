import tensorflow as tf
from discoGAN import DiscoGAN
import utils

print("Building Model")
network = DiscoGAN()
print("Beginning training")
network.train()

