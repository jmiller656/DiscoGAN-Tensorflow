# DiscoGAN for Tensorflow
An implementation of [Learning to Discover Cross-Domain Relations with Generative Adversarial Networks](https://arxiv.org/abs/1703.05192) written in tensorflow.

##Requirements
- Tensorflow 1.0.1
- scipy

##Training
python model.py

##Training details
Make sure your images are under images/DOMAINNAME where domainA and domainB keywords in model.py are your domain names.

##Remarks
As it currently stands, much of the configuration has to be done inside the model.py file. I will soon be making it take command line arguments, download datasets automatically, etc. 
