# Combiner-GANS
Using GANS architecture to solve how broken object did look like before break. "Use cases Archaeology".

#This project is still in gathering data phase 
# Pix2Pix:
Model_2 in models builder represents pix2pix architecture and it doesn't look promsing, so will try with Difusion models

# Data:
So far gathered data of 600 images to 9 objects.

# First Phase:
Gathering 100 images of data at first with some data augmentation, then continue cyclic infrastructure using data versioning to keep track of our data.

Tried to extract feature in this dataset by putting x images and predicting same x images, and wasn't good enough:

 Test Image: 
 
 ![WhatsApp Image 2022-09-24 at 4 44 56 PM (1)](https://user-images.githubusercontent.com/59775002/193263634-e92195c2-7551-4d72-8e50-14255448fcf0.jpeg)

 Predicted Image:
 
![output](https://user-images.githubusercontent.com/59775002/193263530-f41eb919-4ad8-45bb-80d7-5d2098ceb4d7.png)

After training on X and Y, results were:

![output_2](https://user-images.githubusercontent.com/59775002/193263823-887f4751-12c8-4fed-9144-eae954fcb3aa.png)

So will continue gathering data and move with GANs arch.

# Second Phase:
1- Gathering 1000+ pic

2- Use GANs Arch

