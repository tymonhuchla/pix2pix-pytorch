# pix2pix-pytorch
**Table of contents**
* Abstarct
* Networks
* Training
* Abstract

# Abstract

**2016 paper titled <a href='https://arxiv.org/pdf/1611.07004v3.pdf'>Image-to-Image Translation with Conditional Adversarial Networks</a> described an architecture 
called by authors pix2pix. In this repository I brefiely describe main components of this model, share my PyTorch implementation and results.**

**In the paper authors investigate problem of paired image to image translation, and propse an architecture which is able to solve this task.**

The solution is a extension of **GAN** architecture proposed in 2015 by Ian Goodfellow in <a href='https://arxiv.org/pdf/1406.2661.pdf'>this</a> paper. GANs are composed of
a **generator** and a **discriminator** networks which both participate in adversarial fight agaist each other. This way the generator tires to fool the 
discriminator by generating more realistic looking data and the discriminator tires to differentiate real data from fake, produced by the generator. 

Before exploring pix2pix network I recommend a strong read-through <a href='https://arxiv.org/pdf/1406.2661.pdf'>Generative Adversarial Networks</a> paper.

# Architecture

**Generator**

Pix2Pix generator is a model called **U-net** (reference below). U-net is mainly used in image segmentation problems, it has three parts:
* Encoder - takes image as an input and downsamples it
* Bottleneck - connects the encoder and the decoder
* Decoder - upsamples tensor given by the bottleneck
**Image below is a viusal representation of U-net architecture.**

<p align='center'>
  <img src='https://paperswithcode.com/media/methods/Screen_Shot_2020-07-07_at_9.08.00_PM_rpNArED.png'>
</p>

*source https://paperswithcode.com/media/methods/Screen_Shot_2020-07-07_at_9.08.00_PM_rpNArED.png*

**Discriminator**

**PatchGAN** isn't much different from regular GAN. The difference is that regular GANs classify whole image as fake or real while PatchGAN takes *patches* as 
an images and classifies each patch meaning it outputs a matrix of predictions rather than single scalar value.

**PatchGAN visualization.**

<p align='center'>
  <img src='https://www.researchgate.net/profile/Gozde-Unal-2/publication/323904616/figure/fig1/AS:606457334595585@1521602104652/PatchGAN-discriminator-Each-value-of-the-output-matrix-represents-the-probability-of.png'>
</p>

*source https://www.researchgate.net/profile/Gozde-Unal-2/publication/323904616/figure/fig1/AS:606457334595585@1521602104652/PatchGAN-discriminator-Each-value-of-the-output-matrix-represents-the-probability-of.png*

<small><a href='https://arxiv.org/pdf/1505.04597.pdf'>U-net reference<a/> ; <a href='https://arxiv.org/pdf/1611.07004v3.pdf'>PatchGAN reference<a/></small>

# Training

Becouse this project is a part of a bigger project (face image restoration) I have used CelebA dataset. For my target images I have used the regular CelebA images 
and for input images I have just reduced the number of channels from 3 to 1 in the original images. CelebA is included in <code>torch.datasets</code> however I 
prefered to download it myself. You can get the data from my 
<a href='https://drive.google.com/file/d/1n7PHKT7DLkBJHi7yZNGMCXTMNfAohtMO/view?usp=sharing'>google drive</a> or 
<a href='http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html'>official CelebA website</a>.



# Results
