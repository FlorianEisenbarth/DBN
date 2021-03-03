#%%
from numpy.core.arrayprint import _leading_trailing
import pandas as pd
import numpy as np
import os 
import scipy.io
import matplotlib.pyplot as plt 
from math import ceil
#%%
def lire_alpha_digit(caractere):
    
    alphadigit = mat = scipy.io.loadmat('data/binaryalphadigs.mat')["dat"]
    data = {}
    for c in caractere:
        if type(c) == int:
            data[c] = alphadigit[c]
        if type(c) == str:
            data[c] = alphadigit[ord(c.lower())-97 + 10]
    
    return data

def plot_data(data, nb=10):
    if(nb>39): nb=39
    fig, axes = plt.subplots(len(data.keys()), nb, figsize=(10,10))
    k = 0
    for c in data.keys():
        for i in range(0,nb):
            if(len(data.keys())>1):
                axes[k, i].imshow(data[c][i].astype(float), cmap="gray")
        
            else:
                axes[i].imshow(data[c][i].astype(float), cmap="gray")
        k+=1
        
def sigm(X):
    return 1 / (1 + np.exp(-X))

#%%

##################################################################################
#                                     RBM
##################################################################################
class RBM:
    
    def __init__(self,W, a, b) -> None:
        self.W = W
        self.a = a 
        self.b = b 
        
def init_RBM(p,q) -> RBM:
    W = np.random.normal(loc=0, scale=0.1, size=(p,q))
    a = np.zeros(p)
    b = np.zeros(q)
    rbm = RBM(W, a, b)
    return rbm

def entre_sortie_RBM(V: np.ndarray, rbm: RBM) -> np.ndarray:
    p_h_v = sigm(V @ rbm.W + rbm.b)
    return p_h_v

def sortie_entree_RBM(H: np.ndarray, rbm: RBM) -> np.ndarray:
    p_v_h = sigm(H @ rbm.W.T + rbm.a)
    return p_v_h


def train_RBM(X: np.ndarray,rbm :RBM, epochs: int, lr: float, batch_size: int ) -> RBM:
    err_rec = []
    data = X
    for i in range(1,epochs+1):
        np.random.shuffle(X)
        for batch in range(0,len(data),batch_size):
            
            v0 = data[batch:min(batch + batch_size, data.shape[0]),:]
            t = v0.shape[0]
            
            phv0 = entre_sortie_RBM(v0, rbm)
            h0 = (np.random.rand(t, rbm.W.shape[1]) <= phv0).astype(int)
            pvh0 = sortie_entree_RBM(h0, rbm)
            v1 = (np.random.rand(t, rbm.W.shape[0]) <= pvh0).astype(int)
            phv1 = entre_sortie_RBM(v1, rbm)
            
            da = np.sum(v0 - v1, axis=0)
            db = np.sum( phv0 - phv1, axis=0)
            dw = v0.T @ phv0 - v1.T @ phv1
            
            rbm.W += (lr / t) * dw
            rbm.a += (lr / t) * da
            rbm.b += (lr / t) * db
            
        h = entre_sortie_RBM(v0, rbm)
        x_recon = sortie_entree_RBM(h, rbm)
        err_rec.append(np.linalg.norm(v0 - x_recon))
        print("Epochs :{}/{} | Erreur de reconstruction:{}".format(i, epochs ,err_rec[-1]))
    plt.plot(err_rec)
    plt.xlabel("itération")
    plt.ylabel("RMSE")

            
    return rbm, err_rec

def generer_image_RBM(rbm: RBM, iter_gibbs: int, nb_images: int, img_x: int , img_y: int) -> None:
    images = []
    for i in range(nb_images):
        v = (np.random.rand(rbm.W.shape[0]) < 1/2) * 1.0
        for j in range(iter_gibbs):
            ph = entre_sortie_RBM(v, rbm)
            h = (np.random.rand(rbm.W.shape[1]) < ph) * 1.0
            pv = sortie_entree_RBM(h, rbm)
            v = (np.random.rand(rbm.W.shape[0]) < pv) * 1.0
        v = v.reshape((img_y,img_x))
        images.append(v)
    n=0   
    cols = 5
    rows = ceil(nb_images / cols)
    fig, axes = plt.subplots(rows, cols)
    print(axes)
    for j in range(rows):
        for k in range(cols):
            axes[j,k].imshow(images[n], cmap="gray")
            n+=1
            
        
    
    
    
# %%
'''
data = lire_alpha_digit(["A","B"])
data = np.array([i.flatten() for i in np.concatenate(list(data.values()))])

nb_neuro = 100
p = data.shape[1]

rbm = init_RBM(p, nb_neuro)
phv = entre_sortie_RBM(data, rbm)
pvh = sortie_entree_RBM(phv,rbm)
rbm_train = train_RBM(data, rbm, 1000, 0.01, 10)
generer_image_RBM(rbm, 1000, 10, 16, 20)
'''

# %%
