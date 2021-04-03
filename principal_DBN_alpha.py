
##################################################################################
#                                     DBN
##################################################################################
#%%

from principal_RBM_aplha import entre_sortie_RBM, train_RBM, sortie_entree_RBM, lire_alpha_digit, init_RBM
import numpy as np
from math import ceil
import matplotlib.pyplot as plt 

class DNN:
    def __init__(self,layers) -> None:
        self.layers = layers
              
        

def init_DNN(layers) -> DNN:
    dbn = []
    for i in range(0,len(layers)):
        dbn.append(init_RBM(layers[i][0], layers[i][1]))
    
    return DNN(dbn)

def pretrain_DNN(X: np.ndarray, dnn: DNN, epochs: int, lr: float, batch_size: int ) -> DNN:
    
    for i in range(len(dnn.layers)):
        t, _ = train_RBM(X, dnn.layers[i], epochs, lr, batch_size, i)
        dnn.layers[i] = t
        X = entre_sortie_RBM(X,dnn.layers[i])
        
        
    return dnn

def pretrain_DNN_2(X: np.ndarray, dnn: DNN, epochs: int, lr: float, batch_size: int ) -> DNN:
    pretrain_dnn = []
    for i in range(len(dnn.layers)):
        t, _ = train_RBM(X, dnn.layers[i], epochs, lr, batch_size, i)
        pretrain_dnn.append(t)
        X = entre_sortie_RBM(X,t)
        print(t.W)
        
    return DNN(pretrain_dnn)



def generer_image_DBN(dnn: DNN, gibbs_iter: int, nb_images: int, img_x:int, img_y: int) -> None:
    images = []
    for i in range(nb_images):
        v = (np.random.rand(dnn.layers[-1].W.shape[0]) < 1/2 ) * 1.0
        
        for j in range(gibbs_iter):
            ph = entre_sortie_RBM(v, dnn.layers[-1])
            h = (np.random.rand(dnn.layers[-1].W.shape[1]) < ph) * 1.0
            pv = sortie_entree_RBM(h, dnn.layers[-1])
            output_image = (np.random.rand(dnn.layers[-1].W.shape[0]) < pv) * 1.0
        
        for k in range(len(dnn.layers)-2, -1, -1):
            pv = sortie_entree_RBM(output_image, dnn.layers[k])
            output_image = (np.random.rand(dnn.layers[k].W.shape[0]) < pv) * 1.0
            
        output_image = output_image.reshape((img_y,img_x))
        images.append(output_image)   
        
    
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
data = lire_alpha_digit(["F"])
data = np.array([i.flatten() for i in np.concatenate(list(data.values()))])


p = data.shape[1]
layers = [[p,500],[500,300],[300,10]]
img_x = 16
img_y = 20

dnn = init_DNN(layers)
dnn = pretrain_DNN(data, dnn, 250, 0.01, 10)
generer_image_DBN(dnn, 1000, 10, img_x, img_y)
'''

# %%
