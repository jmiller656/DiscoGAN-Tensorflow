# DiscoGAN for Tensorflow
An implementation of [Learning to Discover Cross-Domain Relations with Generative Adversarial Networks](https://arxiv.org/abs/1703.05192) written in tensorflow.

## Requirements
- Tensorflow 1.0.1
- scipy

## Training
`python main.py`

## Training details
Currently the data utils file works on domains from the celeba dataset

## Remarks
As it currently stands, I have refactored much of the model and extracted it to `discoGAN.py`. I will soon be making it take command line arguments, download datasets automatically, etc. As mentioned before, there are now some barebones utilities to work with the celeba dataset.
