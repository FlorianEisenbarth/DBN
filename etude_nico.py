
#%%
from principal_DNN_MNIST import entree_sortie_reseau, retropropagation, test_DNN
from principal_DBN_alpha import init_DNN, pretrain_DNN
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import matplotlib.pyplot as plt 
from math import ceil
from copy import deepcopy
from tensorflow.keras.datasets import mnist

#%%
# Chargement des données de la base MNSIT
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

def analysis_layers():
    # On rend les données flat
    X_train_flatten = np.array([i.flatten() for i in X_train])
    X_test_flatten = np.array([i.flatten() for i in X_test])
    ohe = OneHotEncoder()
    Y_train_one = ohe.fit_transform(Y_train.reshape(-1,1)).toarray()
    Y_test_one = ohe.fit_transform(Y_test.reshape(-1,1)).toarray()

    p = X_train_flatten.shape[1]
    batch_size = 75
    learning_rate = 0.5
    iteration_rbm = 2
    iteration_dnn = 2
    

    for i in range(2, 5):
        neurones = []
        neurones.append((p, 200))
        for j in range(i):
            neurones.append((200, 200))
        neurones.append((200, 10))

        
        DNN_plain = init_DNN(neurones)
        DNN_pretrained = deepcopy(DNN_plain)
        
        DNN_plain = retropropagation(X_train_flatten, Y_train_one, DNN_plain, iteration_dnn, learning_rate, batch_size)  
        DNN_pretrained = pretrain_DNN(X_train_flatten, DNN_pretrained, 2, 0.1, batch_size)
        DNN_pretrained = retropropagation(X_train_flatten, Y_train_one, DNN_pretrained, iteration_dnn, learning_rate, batch_size)
        test_DNN(X_test_flatten, Y_test, DNN_plain)
        test_DNN(X_test_flatten, Y_test, DNN_pretrained)


def analysis_layers_neurons():
    # On rend les données flat
    X_train_flatten = np.array([i.flatten() for i in X_train])
    X_test_flatten = np.array([i.flatten() for i in X_test])
    ohe = OneHotEncoder()
    Y_train_one = ohe.fit_transform(Y_train.reshape(-1,1)).toarray()
    Y_test_one = ohe.fit_transform(Y_test.reshape(-1,1)).toarray()

    p = X_train_flatten.shape[1]
    batch_size = 75
    learning_rate = 0.5
    iteration_rbm = 2
    iteration_dnn = 2
    for i in range(1, 9):
        neurones = [(p, 100*i), (100*i, 100*i), (100*i, 100*i), (100*i, 10)]
        DNN_plain = init_DNN(neurones)
        DNN_pretrained = deepcopy(DNN_plain)
        
        DNN_plain = retropropagation(X_train_flatten, Y_train_one, DNN_plain, iteration_dnn, learning_rate, batch_size)  
        DNN_pretrained = pretrain_DNN(X_train_flatten, DNN_pretrained, 2, 0.1, batch_size)
        DNN_pretrained = retropropagation(X_train_flatten, Y_train_one, DNN_pretrained, iteration_dnn, learning_rate, batch_size)
        test_DNN(X_test_flatten, Y_test, DNN_plain)
        test_DNN(X_test_flatten, Y_test, DNN_pretrained)

def analysis_train_data():
    batch_size = 75
    learning_rate = 0.5
    iteration_rbm = 2
    iteration_dnn = 2
    data_to_train = [1000, 3000, 7000, 10000, 30000, 60000]
    for nb_data in data_to_train:
        # On rend les données flat
        X_train_flatten = np.array([i.flatten() for i in X_train])[0:nb_data]
        X_test_flatten = np.array([i.flatten() for i in X_test])[0:nb_data]
        ohe = OneHotEncoder()
        Y_train_one = ohe.fit_transform(Y_train.reshape(-1,1)).toarray()[0:nb_data]
        Y_test_one = ohe.fit_transform(Y_test.reshape(-1,1)).toarray()[0:nb_data]

        DNN_plain = init_DNN(neurones)
        DNN_pretrained = deepcopy(DNN_plain)
        
        DNN_plain = retropropagation(X_train_flatten, Y_train_one, DNN_plain, iteration_dnn, learning_rate, batch_size)  
        DNN_pretrained = pretrain_DNN(X_train_flatten, DNN_pretrained, 2, 0.1, batch_size)
        DNN_pretrained = retropropagation(X_train_flatten, Y_train_one, DNN_pretrained, iteration_dnn, learning_rate, batch_size)
        test_DNN(X_test_flatten, Y_test, DNN_plain)
        test_DNN(X_test_flatten, Y_test, DNN_pretrained)
# %%
analysis_layers()
analysis_layers_neurons()
analysis_train_data()
# %%
