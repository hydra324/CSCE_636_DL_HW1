#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 12:00:48 2019

@author: 
"""

from cProfile import label
from random import shuffle
import numpy as np
import sys

"""This script implements a two-class logistic regression model.
"""

class logistic_regression_multiclass(object):
	
    def __init__(self, learning_rate, max_iter, k):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.k = k 
        
    def fit_miniBGD(self, X, labels, batch_size):
        """Train perceptron model on data (X,y) with mini-Batch GD.

        Args:
            X: An array of shape [n_samples, n_features].
            labels: An array of shape [n_samples,].  Only contains 0,..,k-1.
            batch_size: An integer.

        Returns:
            self: Returns an instance of self.

        Hint: the labels should be converted to one-hot vectors, for example: 1----> [0,1,0]; 2---->[0,0,1].
        """

		### YOUR CODE HERE
        n_samples, n_features = X.shape
        # convert labels into one hot vectors
        Y = np.zeros(shape=(n_samples,self.k))
        Y[np.arange(n_samples),labels.astype(int)]=1

        # initialize weights to zero or randomize
        # self.W = np.random.rand(n_features,self.k)
        self.W = np.zeros(shape=(n_features,self.k))

        print("")
        for epoch in range(self.max_iter):
            # begin epoch

            # print training progress
            sys.stdout.write('\r')
            sys.stdout.write("Training...epoch=%i"%(epoch))
            sys.stdout.flush()

            # shuffle data and labels in unison
            indices = np.arange(n_samples)
            np.random.shuffle(indices)
            X = X[indices]
            Y = Y[indices]

            b_start = 0 # batch start (inclusive)
            b_end = min(b_start+batch_size,n_samples) # batch end (exclusive)
            while b_start < n_samples:
                X_batch,Y_batch = X[b_start:b_end],Y[b_start:b_end]
                grad = 0
                for sample in range(X_batch.shape[0]):
                    _x,_y = X_batch[sample],Y_batch[sample]
                    # compute gradient using the equation grad L w.r.t W = _x @ (p-_y).T
                    grad += self._gradient(_x,_y)
                grad = grad/batch_size
                # update weights
                self.W = self.W - self.learning_rate * grad
                # update batch start and end
                b_start = b_end
                b_end = min(b_start+batch_size,n_samples)
            # end epoch
        print("")

		### END YOUR CODE
        return self
    
    def fit_miniBGD_print_weights(self, X, labels, batch_size):
        """ Same as the above method for mini-Batch GD, except the it will print weights at the end of each epoch.
        Train perceptron model on data (X,y) with mini-Batch GD.

        Args:
            X: An array of shape [n_samples, n_features].
            labels: An array of shape [n_samples,].  Only contains 0,..,k-1.
            batch_size: An integer.

        Returns:
            self: Returns an instance of self.

        Hint: the labels should be converted to one-hot vectors, for example: 1----> [0,1,0]; 2---->[0,0,1].
        """

        n_samples, n_features = X.shape
        # convert labels into one hot vectors
        Y = np.zeros(shape=(n_samples,self.k))
        Y[np.arange(n_samples),labels.astype(int)]=1

        # initialize weights to zero or randomize
        # self.W = np.random.rand(n_features,self.k)
        self.W = np.zeros(shape=(n_features,self.k))

        print("")
        for epoch in range(self.max_iter):
            # begin epoch

            # print training progress
            sys.stdout.write('\r')
            sys.stdout.write("Training...epoch=%i"%(epoch))
            sys.stdout.flush()

            # shuffle data and labels in unison
            indices = np.arange(n_samples)
            np.random.shuffle(indices)
            X = X[indices]
            Y = Y[indices]

            b_start = 0 # batch start (inclusive)
            b_end = min(b_start+batch_size,n_samples) # batch end (exclusive)
            while b_start < n_samples:
                X_batch,Y_batch = X[b_start:b_end],Y[b_start:b_end]
                grad = 0
                for sample in range(X_batch.shape[0]):
                    _x,_y = X_batch[sample],Y_batch[sample]
                    # compute gradient using the equation grad L w.r.t W = _x @ (p-_y).T
                    grad += self._gradient(_x,_y)
                grad = grad/batch_size
                # update weights
                self.W = self.W - self.learning_rate * grad
                # update batch start and end
                b_start = b_end
                b_end = min(b_start+batch_size,n_samples)
            # end epoch
            print("after epoch=%i, softmax weights=\n"%(epoch),self.W)
        print("")


        return self
    

    def _gradient(self, _x, _y):
        """Compute the gradient of cross-entropy with respect to self.W
        for one training sample (_x, _y). This function is used in fit_*.

        Args:
            _x: An array of shape [n_features,].
            _y: One_hot vector. 

        Returns:
            _g: An array of shape [n_features, k]. The gradient of
                cross-entropy with respect to self.W.
        """
		### YOUR CODE HERE
        # grad L w.r.t a_k = p_k-a_k, where p_k is e^(w_kT.x)/sigma(e^w_kT.x)
        # grad L w.r.t W = [(p1-y1)_x,(p2-y2)_x,...,(pk-yk)_x]
        # (p-y).T = [p1-y1,p2-y2,..,pk-yk]
        # which means grad L w.r.t W = _x @ (p-y).T
        p = self.softmax(self.W.T @ _x) # output of softmax layer
        _g = _x.reshape(-1,1) @ (p-_y).reshape(-1,1).T
		### END YOUR CODE
        return _g
    
    def softmax(self, x):
        """Compute softmax values for each sets of scores in x."""
        ### You must implement softmax by youself, otherwise you will not get credits for this part.

		### YOUR CODE HERE
        # np.exp throws overflow error when we have pontially large values
        # to overcome this, we subtract max(x) from x
        # the result will still be the same
        # because softmax(x) = softmax(x-k) can be proven mathematically
        z = x-max(x)
        p = np.exp(z)/np.sum(np.exp(z))
		### END YOUR CODE
        return p
    
    def get_params(self):
        """Get parameters for this perceptron model.

        Returns:
            W: An array of shape [n_features, k].
        """
        if self.W is None:
            print("Run fit first!")
            sys.exit(-1)
        return self.W


    def predict(self, X):
        """Predict class labels for samples in X.

        Args:
            X: An array of shape [n_samples, n_features].

        Returns:
            preds: An array of shape [n_samples,]. Only contains 0,..,k-1.
        """
		### YOUR CODE HERE
        # X.shape: (n_samples,n_feats) ,W.shape:(n_feats,k)
        # p = n_samples,k
        p = np.array([self.softmax(self.W.T @ _x) for _x in X]) # predicted probablilities
        preds = np.argmax(p,axis=1)
		### END YOUR CODE
        return preds


    def score(self, X, labels):
        """Returns the mean accuracy on the given test data and labels.

        Args:
            X: An array of shape [n_samples, n_features].
            labels: An array of shape [n_samples,]. Only contains 0,..,k-1.

        Returns:
            score: An float. Mean accuracy of self.predict(X) wrt. labels.
        """
		### YOUR CODE HERE
        preds = self.predict(X)
        score = np.sum(np.equal(preds,labels)) / len(labels)
		### END YOUR CODE
        return score

