# Combiner-GANS
Using GANS architecture to solve how broken object did look like before break. "Use cases Archaeology".

# This project is still in progress.... 

# Pix2Pix:
Model_2 in models builder represents pix2pix architecture and it doesn't look promsing, so will try with Diffusion models

# Data:
So far gathered data of 600 images to 9 objects.

# First Phase:
Gathering 100 images of data at first with some data augmentation, then continue cyclic infrastructure using data versioning to keep track of our data.

Tried to extract feature in this dataset by putting x images and predicting same x images, and wasn't good enough:

 Test Image: 
 
 ![WhatsApp Image 2022-09-24 at 4 44 56 PM (1)](https://user-images.githubusercontent.com/59775002/193263634-e92195c2-7551-4d72-8e50-14255448fcf0.jpeg)
 
 Predicted Image "Pix2pix"("60 epochs", "Data_version: 3-10-22"):

![image](https://user-images.githubusercontent.com/59775002/196940869-293778f4-8155-4dc7-9994-58ca26ac35c6.png)

 Predicted Image "AUTOENCODER":

![output_2](https://user-images.githubusercontent.com/59775002/193263823-887f4751-12c8-4fed-9144-eae954fcb3aa.png)

So will continue gathering data and move with GANs arch.



