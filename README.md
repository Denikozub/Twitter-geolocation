# Twitter users geolocation based on tweet texts

![image](https://user-images.githubusercontent.com/41386672/173209011-3367e639-67b6-489f-982e-a592144b4206.png)
<img src=https://user-images.githubusercontent.com/41386672/173209108-293624e3-463f-47b8-9958-2e2969ec734c.png alt="" height="200" width="150"/>
<img src=https://user-images.githubusercontent.com/41386672/173209155-848326ca-d368-452c-a69f-327a433cbee3.png alt="" hspace="20" width="200"/>

## Data
Processed [dataset](https://archive.org/details/twitter_cikm_2010) contains 620k tweets and corresponding coordinates.  
Processing includes geocoding US cities, which is done using Nominatim, and country location check.  
Train - test - val split: 80% - 10% - 10%, batch size = 64.  
The task is to predict coorditates (lat - lon) based on tweet texts.

## Loss function
To estimate distance betweet predicted and real coordinates, haversine distance is used.  
It considers Earth as a sphere with a set radius, which is its simplest representation.  
![image](https://user-images.githubusercontent.com/41386672/173208972-6269ed60-f87b-4325-87fd-8558a0d6e9cd.png)

## Models

### Baseline model
* BERT tokenizer is used with max_length=32, truncation=True
* Takes use of BERT \<CLS\> token embeddings only
* They are fed to two linear layers, followed up by linear regression
* Each layer uses batch normalization
* ReLU is used as an activation function

### Autoencoder model
* Used for dimensionality reduction
* Denoising architecture (with scalable factor)
* BERT weights are disabled while training AE
* MSE loss is used for autoencoder training
* Both encoder and decoder consist of two layers with ReLU activation
* Encoder states are saved during training and used in regression model

<img src=https://user-images.githubusercontent.com/41386672/173209765-9bedb5d1-1aaa-479e-a075-defcfcd29cff.png alt="" width="400"/>
