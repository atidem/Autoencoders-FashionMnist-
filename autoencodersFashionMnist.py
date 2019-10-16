# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 14:49:47 2019

@author: atidem
"""
from keras.models import Model
from keras.layers import Input,Dense
from keras.datasets import fashion_mnist
import matplotlib.pyplot as plt 
import json,codecs 

#%% load data and preprocess

(xTrain,_),(xTest,_) = fashion_mnist.load_data() 

xTrain = xTrain.astype("float32") / 255.0
xTest = xTest.astype("float32") / 255.0

xTrain = xTrain.reshape((len(xTrain),xTrain.shape[1:][0]*xTrain.shape[1:][1]))
xTest = xTest.reshape((len(xTest),xTest.shape[1:][0]*xTest.shape[1:][1]))

#%% visualize
plt.imshow(xTrain[89].reshape(28,28))
plt.axis("off")
plt.show()

#%% model and train

inputImg = Input(shape=(784,))

encoded = Dense(32,activation="relu")(inputImg)  # bir önceki ile current place holderı birbirine bağladık..

encoded = Dense(16,activation="relu")(encoded)

decoded = Dense(32,activation="relu")(encoded)

output = Dense(784,activation="sigmoid")(decoded)

autoencoder = Model(inputImg,output)

autoencoder.compile(optimizer="rmsprop",loss="binary_crossentropy")

hist = autoencoder.fit(xTrain,xTrain,epochs=200,
                       batch_size=256,shuffle=True,
                       validation_data=(xTrain,xTrain))


#%% save model 
autoencoder.save_weights("autoencodersFashionMnist.h5")

#%%

print(hist.history.keys())
plt.plot(hist.history["loss"],label="training loss")
plt.plot(hist.history["val_loss"],label="test loss")
plt.legend()
plt.show()

#%% 
import json,codecs
with open("scores.json","w") as f:
    json.dump(hist.history,f)

#%%
with codecs.open("scores.json","r",encoding="utf-8") as f:
    scores = json.loads(f.read())

plt.plot(scores["loss"],label="training loss")
plt.plot(scores["val_loss"],label="test loss")
plt.legend()
plt.show()

#%% buldugu featurelar

encoder = Model(inputImg,encoded)
encodedImg = encoder.predict(xTest)

plt.imshow(xTest[1500].reshape(28,28))
plt.axis("off")
plt.show()
plt.figure()
plt.imshow(encodedImg[1500].reshape(4,4))
plt.axis("off")
plt.show()


#%% girdi ve çıktılar 
decImg = autoencoder.predict(xTest)

n = 10 
plt.figure(figsize=(20,4))
for i in range(n):
    ax=plt.subplot(2,n,i+1)
    plt.imshow(xTest[i].reshape(28,28))
    plt.axis("off")
    
    ax=plt.subplot(2,n,i+1+n)
    plt.imshow(decImg[i].reshape(28,28))
    plt.axis("off")

plt.show()
    
