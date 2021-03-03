#%%
from numpy.core.numeric import cross
from principal_DBN_alpha import *
from principal_RBM_aplha import *
import numpy as np
from copy import deepcopy
from tensorflow.keras.datasets import mnist
from sklearn.preprocessing import OneHotEncoder


#%%
def softmax(X: np.ndarray) -> np.ndarray:
    return np.exp(X)/np.sum(np.exp(X), axis=1).reshape(-1,1)


def calcul_softmax(data: np.ndarray, rbm: RBM) -> np.ndarray:
    p = data @ rbm.W + rbm.b 
    p = softmax(p)
    return p


def entree_sortie_reseau(data: np.ndarray, dnn: DNN) -> np.ndarray:
    sortie = []
    for i in range(len(dnn.layers)-1):
        a = entre_sortie_RBM(data, dnn.layers[i])
        data = a
        sortie.append(a)
    sortie.append(calcul_softmax(data ,dnn.layers[-1]))
    
    return sortie


def retropropagation(X: np.ndarray, Y:np.ndarray , dnn: DNN, iter: int, lr: float, batch_size: int, ) -> DNN:
    
    cross_entropy = []
    indices = np.arange(0,X.shape[0])
    np.random.shuffle(indices)
    
    for i in range(1, iter+1):
        np.random.shuffle(X)
        for batch in range(0, X.shape[0], batch_size):
            batch_index = indices[batch:min(batch + batch_size, X.shape[0])] 
            data_batch = X[batch_index,:]
            dnn_copy = deepcopy(dnn)
            
            sortie = entree_sortie_reseau(data_batch, dnn)
            
            c = sortie[-1] - Y[batch_index]
            d_wp = sortie[-2].T @ c
            d_bp = c.sum(0)
            dnn_copy.layers[-1].W -= (lr/data_batch.shape[0]) * d_wp
            dnn_copy.layers[-1].b -= (lr/data_batch.shape[0]) * d_bp
            
            for j in range(len(dnn.layers)-2,-1,-1):
                if(j == 0):
                    x = data_batch
                else:
                    x = sortie[j-1]

                c_mul = sortie[j] * ( 1 - sortie[j])
                c = (c @ dnn.layers[j+1].W.T) * c_mul
                d_wp = x.T @ c 
                d_bp = c.sum(0)
                dnn_copy.layers[j].W -= (lr/data_batch.shape[0]) * d_wp
                dnn_copy.layers[j].b -= (lr/data_batch.shape[0]) * d_bp
        
            dnn = deepcopy(dnn_copy)
    
        sortie = entree_sortie_reseau(X, dnn)
        erreur = - (np.log10(sortie[-1]) * (Y==1)*1).sum()
        cross_entropy.append(erreur)
        print("Epochs :{}/{} | cross-entropy:{}".format(i, iter ,cross_entropy[-1]))
    
    plt.figure(figsize=(10, 7))
    plt.plot(cross_entropy)
    plt.legend(['Entropie croisée'])
    plt.title("Entropie croisée en fonctions des iterations")
    plt.xlabel("itérations")
    plt.ylabel('entropie croisée')
    return dnn
        

def test_DNN(X_test, Y_test, dnn):
    pred = entree_sortie_reseau(X_test,dnn)
    erreur = - (np.log10(pred[-1][0]) * (Y_test==1)*1).sum()
    print("Categorical-cross-entropy:{}".format(erreur))
    
                
        
# %%

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

X_train = np.array([i.flatten() for i in X_train]) 
X_test = np.array([i.flatten() for i in X_test])

X_train = np.array([(i/256>0.5).astype('int') for i in X_train])[0:10000]


ohe = OneHotEncoder()

Y_train_one = ohe.fit_transform(Y_train.reshape(-1,1)).toarray()[0:10000]
Y_test_one = ohe.fit_transform(Y_test.reshape(-1,1)).toarray()

p = X_train.shape[1]
layers = [(p, 300), (300, 200), (200, 10)]
img_x = 16
img_y = 20

dnn = init_DNN(layers)
#dnn = pretrain_DNN(X_train, dnn, 1000, 0.01, 10)
dnn = retropropagation(X_train, Y_train_one, dnn, 250, 0.16, 75)


# %%
