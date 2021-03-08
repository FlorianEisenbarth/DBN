#%%
from principal_DNN_MNIST import entree_sortie_reseau, retropropagation
from principal_DBN_alpha import init_DNN, pretrain_DNN
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import matplotlib.pyplot as plt 
from math import ceil
from tensorflow.keras.datasets import mnist

#%%
# Chargement des données de la base MNSIT
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()


# On rend les données flat
X_train_flatten = np.array([i.flatten() for i in X_train])
X_test_flatten = np.array([i.flatten() for i in X_test])

X_train_flatten = np.array([(i/256>0.5).astype('int') for i in X_train_flatten])[0:10000]
X_test_flatten = np.array([(i/256>0.5).astype('int') for i in X_test_flatten])[0:10000]

#OneHot encoding des targets 
ohe = OneHotEncoder()
Y_train_one = ohe.fit_transform(Y_train.reshape(-1,1)).toarray()[0:10000]
Y_test_one = ohe.fit_transform(Y_test.reshape(-1,1)).toarray()[0:10000]


# Paramètres d'apprentissage
p = X_train_flatten.shape[1]
neurones = [[p,300],[300,200],[200,10]]
batch_size = 75
learning_rate = 0.5
iteration_rbm = 100
iteration_dnn = 250

# reseeaux 
Dnn1 = init_DNN(neurones)
Dnn2 = init_DNN(neurones)

# pretrain de Dnn1
pre_train_Dnn1 = pretrain_DNN(X_train_flatten, Dnn1, 10, 0.1, batch_size)

#appretissage des reseaux
Dnn2 = retropropagation(X_train_flatten, Y_train_one, Dnn2, iteration_dnn, learning_rate, batch_size)  
Dnn1 = retropropagation(X_train_flatten, Y_train_one, pre_train_Dnn1, iteration_dnn, learning_rate, batch_size)


# %%

#predictions 
pred1 = entree_sortie_reseau(X_test_flatten, Dnn1)
pred1 = pred1[-1].argmax(axis=1)

pred2 = entree_sortie_reseau(X_test_flatten, Dnn2)
pred2 = pred2[-1].argmax(axis=1)

err1 = 1 - (( pred1 == Y_test).astype(int).sum(axis=0) / len(X_test)) 
err2 = 1 - (( pred2 == Y_test).astype(int).sum(axis=0) / len(X_test)) 

print("Taux d'erreur de classification Dnn1(pre entraine):{} %".format(err1 *100))
print("Taux d'erreur de classification Dnn1(pre entraine):{} %".format(err2 *100))
# %%
