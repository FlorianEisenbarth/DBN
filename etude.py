#%%
from principal_DNN_MNIST import entree_sortie_reseau, retropropagation
from principal_DBN_alpha import init_DNN, pretrain_DNN
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import matplotlib.pyplot as plt 
from math import ceil
from tensorflow.keras.datasets import mnist



#%%
np.random.seed(42)
# Chargement des données de la base MNSIT
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()


# On rend les données flat
X_train_flatten = np.array([i.flatten() for i in X_train])
X_test_flatten = np.array([i.flatten() for i in X_test])

X_train_flatten = np.array([(i/256>0.5).astype('int') for i in X_train_flatten])
X_test_flatten = np.array([(i/256>0.5).astype('int') for i in X_test_flatten])

#OneHot encoding des targets 
ohe = OneHotEncoder()
Y_train_one = ohe.fit_transform(Y_train.reshape(-1,1)).toarray()
Y_test_one = ohe.fit_transform(Y_test.reshape(-1,1)).toarray()


# Paramètres d'apprentissage
p = X_train_flatten.shape[1]

#neurones = [[p,200],[200,10]]
#neurones = [[p,200],[200,200],[200,10]]
#neurones = [[p,200],[200,200],[200,200],[200,10]]
#neurones = [[p,200],[200,200],[200,200],[200,200],[200,10]]
neurones = [[p,200],[200,200],[200,200],[200,200],[200,200],[200,10]]

batch_size = 75
learning_rate = 0.5
learning_rate_rbm = 0.01
iteration_rbm = 100
iteration_dnn = 200

# reseeaux 
Dnn1 = init_DNN(neurones)
#Dnn2 = init_DNN(neurones)

# pretrain de Dnn1
pre_train_Dnn1 = pretrain_DNN(X_train_flatten, Dnn1, iteration_rbm, learning_rate_rbm, batch_size)

#appretissage des reseaux
Dnn1 = retropropagation(X_train_flatten, Y_train_one, pre_train_Dnn1, iteration_dnn, learning_rate, batch_size)
#Dnn2 = retropropagation(X_train_flatten, Y_train_one, Dnn2, iteration_dnn, learning_rate, batch_size)  

# %%

#predictions 
pred1 = entree_sortie_reseau(X_test_flatten, Dnn1)
pred1 = pred1[-1].argmax(axis=1)

#pred2 = entree_sortie_reseau(X_test_flatten, Dnn2)
#pred2 = pred2[-1].argmax(axis=1)

err1 = 1 - (( pred1 == Y_test).astype(int).sum(axis=0) / len(X_test)) 
#err2 = 1 - (( pred2 == Y_test).astype(int).sum(axis=0) / len(X_test)) 

print("Taux d'erreur de classification Dnn1(pre entraine):{} %".format(err1 *100))
#print("Taux d'erreur de classification Dnn1(pre entraine):{} %".format(err2 *100))

couches = range(1,6)
cross_entropy_pretrain = [23.55, 11.21, 6.00, 4.38, 2.89]
cross_entropy_nonpretrain = [13.93, 3.34, 1.49, 0.87, 0.57]
err_pretrain = [ 3.49, 2.73, 2.28, 2.05, 1.96 ]
err_nonpretrain = [2.20, 2.15, 2.54, 2.52, 3.06]

plt.figure()
plt.plot(couches,cross_entropy_pretrain, label="pretrain")
plt.plot(couches, cross_entropy_nonpretrain, label="non pretrain")
plt.legend()
plt.title("Evolution de l'entropie croisé en fonction du nombres de couches cachées ")
plt.xlabel("couches cachées ")
plt.xticks(range(1,6))
plt.ylabel("Entropie croisé")

plt.figure()
plt.plot(couches,err_pretrain, label="pretrain")
plt.plot(couches,err_nonpretrain, label="non pretrain")
plt.legend()
plt.title("Evolution de l'erreur en fonction du nombres de couches cachées")
plt.xlabel("couches cachées ")
plt.xticks(range(1,6))
plt.ylabel("erreur (%)")
