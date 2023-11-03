# -*- coding: utf-8 -*-

######
# Timothée Blanchy (timb1101)
# Oussama Khaloui (khao1201)
####

import numpy as np
from sklearn.linear_model import Perceptron
import matplotlib.pyplot as plt


class ClassifieurLineaire:
    def __init__(self, lamb, methode):
        """
        Algorithmes de classification lineaire

        L'argument ``lamb`` est une constante pour régulariser la magnitude
        des poids w et w_0

        ``methode`` :   1 pour classification generative
                        2 pour Perceptron
                        3 pour Perceptron sklearn
        """
        self.w = np.array([1., 2.]) # paramètre aléatoire
        self.w_0 = -5.              # paramètre aléatoire
        self.lamb = lamb
        self.methode = methode

    def entrainement(self, x_train, t_train):
        """
        Entraîne deux classifieurs sur l'ensemble d'entraînement formé des
        entrées ``x_train`` (un tableau 2D Numpy) et des étiquettes de classe cibles
        ``t_train`` (un tableau 1D Numpy).

        Lorsque self.method = 1 : implémenter la classification générative de
        la section 4.2.2 du libre de Bishop. Cette méthode doit calculer les
        variables suivantes:

        - ``p`` scalaire spécifié à l'équation 4.73 du livre de Bishop.

        - ``mu_1`` vecteur (tableau Numpy 1D) de taille D, tel que spécifié à
                    l'équation 4.75 du livre de Bishop.

        - ``mu_2`` vecteur (tableau Numpy 1D) de taille D, tel que spécifié à
                    l'équation 4.76 du livre de Bishop.

        - ``sigma`` matrice de covariance (tableau Numpy 2D) de taille DxD,
                    telle que spécifiée à l'équation 4.78 du livre de Bishop,
                    mais à laquelle ``self.lamb`` doit être ADDITIONNÉ À LA
                    DIAGONALE (comme à l'équation 3.28).

        - ``self.w`` un vecteur (tableau Numpy 1D) de taille D tel que
                    spécifié à l'équation 4.66 du livre de Bishop.

        - ``self.w_0`` un scalaire, tel que spécifié à l'équation 4.67
                    du livre de Bishop.

        lorsque method = 2 : Implementer l'algorithme de descente de gradient
                        stochastique du perceptron avec 1000 iterations

        lorsque method = 3 : utiliser la librairie sklearn pour effectuer une
                        classification binaire à l'aide du perceptron

        """
        if self.methode == 1:  # Classification generative
            print('Classification generative')
                    
            N = len(t_train)  # nombre de données
            N1 = np.sum(t_train)  # nombre de données de classe 1
            N2 = N - N1  # nombre de données de classe 2
                    
            p = N1 / N

            mu_1 = np.sum(t_train[:, np.newaxis] * x_train, axis=0) / N1
            mu_2 = np.sum((1 - t_train[:, np.newaxis]) * x_train, axis=0) / N2
                    
            x_t_1 = x_train - mu_1
            x_t_2 = x_train - mu_2
                    
            sigma1_matrices = x_t_1[:, :, np.newaxis] * x_t_1[:, np.newaxis, :]
            sigma2_matrices = x_t_2[:, :, np.newaxis] * x_t_2[:, np.newaxis, :]
                    
            sigma1 = np.sum(sigma1_matrices * t_train[:, np.newaxis, np.newaxis], axis=0) / N1
            sigma2 = np.sum(sigma2_matrices * (1 - t_train[:, np.newaxis, np.newaxis]), axis=0) / N2
                    
            sigma = p * sigma1 + (1 - p) * sigma2 + self.lamb * np.eye(2)
            
            sigma_inv = np.linalg.inv(sigma)

            self.w = np.dot(sigma_inv, (mu_1 - mu_2))

            self.w_0 = -0.5 * np.dot(np.dot(mu_1.T, sigma_inv), mu_1) \
                    + 0.5 * np.dot(np.dot(mu_2.T, sigma_inv), mu_2) \
                    + np.log(p / (1-p))




        elif self.methode == 2:  # Perceptron + SGD, learning rate = 0.001, nb_iterations_max = 1000
            print('Perceptron')

            # On initialise les paramètres
            self.w = np.random.rand(x_train.shape[1])
            self.w_0 = np.random.rand()
            learning_rate = 0.001
            nb_iterations_max = 1000

            # On effectue la descente de gradient
            for iter in range(nb_iterations_max):
                for i in range(t_train.shape[0]):
                    # Actual output
                    z = np.dot(self.w, x_train[i]) + self.w_0
                    z_act = 1 if z >= 0 else 0

                    # Update weights
                    self.w += learning_rate * ((t_train[i] - z_act) * x_train[i])
                    self.w_0 += learning_rate * (t_train[i] - z_act)


        else:  # Perceptron + SGD [sklearn] + learning rate = 0.001 + penalty 'l2' voir http://scikit-learn.org/
            print('Perceptron [sklearn]')
            # On utilise le perceptron de sklearn
            perceptron = Perceptron(penalty='l2', alpha=self.lamb, max_iter=1000, eta0=0.001)
            perceptron.fit(x_train, t_train)
            self.w = perceptron.coef_[0]
            self.w_0 = perceptron.intercept_[0]


        print('w = ', self.w, 'w_0 = ', self.w_0, '\n')

    def prediction(self, x):
        """
        Retourne la prédiction du classifieur lineaire.  Retourne 1 si x est
        devant la frontière de décision et 0 sinon.

        ``x`` est un tableau 1D Numpy

        Cette méthode suppose que la méthode ``entrainement()``
        a préalablement été appelée. Elle doit utiliser les champs ``self.w``
        et ``self.w_0`` afin de faire cette classification.
        """
        return int(np.dot(self.w, x) + self.w_0 > 0)

    @staticmethod
    def erreur(t, prediction):
        """
        Retourne l'erreur de classification, i.e.
        1. si la cible ``t`` et la prédiction ``prediction``
        sont différentes, 0. sinon.
        """
        return int(t != prediction)

    def afficher_donnees_et_modele(self, x_train, t_train, x_test, t_test):
        """
        afficher les donnees et le modele

        x_train, t_train : donnees d'entrainement
        x_test, t_test : donnees de test
        """
        plt.figure(0)
        plt.scatter(x_train[:, 0], x_train[:, 1], s=t_train * 100 + 20, c=t_train)

        pente = -self.w[0] / self.w[1]
        xx = np.linspace(np.min(x_test[:, 0]) - 2, np.max(x_test[:, 0]) + 2)
        yy = pente * xx - self.w_0 / self.w[1]
        plt.plot(xx, yy)
        plt.title('Training data')

        plt.figure(1)
        plt.scatter(x_test[:, 0], x_test[:, 1], s=t_test * 100 + 20, c=t_test)

        pente = -self.w[0] / self.w[1]
        xx = np.linspace(np.min(x_test[:, 0]) - 2, np.max(x_test[:, 0]) + 2)
        yy = pente * xx - self.w_0 / self.w[1]
        plt.plot(xx, yy)
        plt.title('Testing data')

        plt.show()

    def parametres(self):
        """
        Retourne les paramètres du modèle
        """
        return self.w_0, self.w
