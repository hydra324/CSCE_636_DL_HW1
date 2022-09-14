from turtle import shape
import numpy as np
import sys

"""This script implements a two-class logistic regression model.
"""

class logistic_regression(object):
	
    def __init__(self, learning_rate, max_iter):
        self.learning_rate = learning_rate
        self.max_iter = max_iter

    def fit_BGD(self, X, y):
        """Train perceptron model on data (X,y) with Batch Gradient Descent.

        Args:
            X: An array of shape [n_samples, n_features].
            y: An array of shape [n_samples,]. Only contains 1 or -1.

        Returns:
            self: Returns an instance of self.
        """
        n_samples, n_features = X.shape

		### YOUR CODE HERE
        n_features = X.shape[1]
        self.W = np.zeros(shape=(n_features,))
        for epoch in range(self.max_iter):
            grad = 0
            for sample in range(n_samples):
                grad += self._gradient(X[sample],y[sample])
            grad = grad/n_samples
            self.W = self.W - self.learning_rate * grad
		### END YOUR CODE
        return self

    def fit_miniBGD(self, X, y, batch_size):
        """Train perceptron model on data (X,y) with mini-Batch Gradient Descent.

        Args:
            X: An array of shape [n_samples, n_features].
            y: An array of shape [n_samples,]. Only contains 1 or -1.
            batch_size: An integer.

        Returns:
            self: Returns an instance of self.
        """
		### YOUR CODE HERE
        n_features = X.shape[1]
        self.W = np.zeros(shape=(n_features,))

        for epoch in range(self.max_iter):
            # begin epoch
            # shuffle data and labels in unision
            n_samples = y.shape[0]
            indices = np.arange(n_samples)
            np.random.shuffle(indices)
            X = X[indices]
            y = y[indices]

            b_start = 0 # batch start (inclusive)
            b_end = min(b_start+batch_size,n_samples) # batch end (exclusive)
            while b_start<n_samples:
                X_batch,y_batch = X[b_start:b_end],y[b_start:b_end]
                grad = 0
                for sample in range(X_batch.shape[0]):
                    _x,_y = X_batch[sample],y_batch[sample]
                    # compute gradient using the equation grad(E(w)) = -yx_n/(1+e^ywT.x_n)
                    grad += self._gradient(_x,_y)
                grad = grad/batch_size
                # update weights
                self.W = self.W - self.learning_rate * grad
                # update batch start and end
                b_start = b_end
                b_end = min(b_start+batch_size,n_samples)
            # end epoch

		### END YOUR CODE
        return self

    def fit_SGD(self, X, y):
        """Train perceptron model on data (X,y) with Stochastic Gradient Descent.

        Args:
            X: An array of shape [n_samples, n_features].
            y: An array of shape [n_samples,]. Only contains 1 or -1.

        Returns:
            self: Returns an instance of self.
        """
		### YOUR CODE HERE
        n_features = X.shape[1]
        self.W = np.zeros(shape=(n_features,))        
        for epoch in range(self.max_iter):
            # begin epoch
            # shuffle data and labels in unision
            n_samples = y.shape[0]
            indices = np.arange(n_samples)
            np.random.shuffle(indices)
            X = X[indices]
            y = y[indices]

            for sample in range(n_samples):
                _x,_y = X[sample],y[sample]
                # compute gradient using the equation grad(E(w)) = -yx_n/(1+e^ywT.x_n)
                _g = self._gradient(_x,_y)
                # update weights
                self.W = self.W - self.learning_rate * _g
            # end epoch

		### END YOUR CODE
        return self

    def _gradient(self, _x, _y):
        """Compute the gradient of cross-entropy with respect to self.W
        for one training sample (_x, _y). This function is used in fit_*.

        Args:
            _x: An array of shape [n_features,].
            _y: An integer. 1 or -1.

        Returns:
            _g: An array of shape [n_features,]. The gradient of
                cross-entropy with respect to self.W.
        """
		### YOUR CODE HERE
        # grad(E(w)) = -yx/(1+e^ywT.x)
        _g = -_y * _x / (1+np.exp(_y*np.dot(self.W.T,_x)))
		### END YOUR CODE
        return _g

    def get_params(self):
        """Get parameters for this perceptron model.

        Returns:
            W: An array of shape [n_features,].
        """
        if self.W is None:
            print("Run fit first!")
            sys.exit(-1)
        return self.W

    def predict_proba(self, X):
        """Predict class probabilities for samples in X.

        Args:
            X: An array of shape [n_samples, n_features].

        Returns:
            preds_proba: An array of shape [n_samples, 2].
                Only contains floats between [0,1].
        """
		### YOUR CODE HERE
        pos_prob = 1/(1+np.exp(-X @ self.W))
        neg_prob = 1-pos_prob
        preds_proba = np.concatenate((pos_prob,neg_prob),axis=1)
		### END YOUR CODE
        return preds_proba


    def predict(self, X):
        """Predict class labels for samples in X.

        Args:
            X: An array of shape [n_samples, n_features].

        Returns:
            preds: An array of shape [n_samples,]. Only contains 1 or -1.
        """
		### YOUR CODE HERE
        preds = X @ self.W
        preds[np.where(preds>=0)]=1
        preds[np.where(preds<0)]=-1
		### END YOUR CODE
        return preds

    def score(self, X, y):
        """Returns the mean accuracy on the given test data and labels.

        Args:
            X: An array of shape [n_samples, n_features].
            y: An array of shape [n_samples,]. Only contains 1 or -1.

        Returns:
            score: An float. Mean accuracy of self.predict(X) wrt. y.
        """
		### YOUR CODE HERE
        preds = self.predict(X)
        score  = np.sum(np.equal(y, preds)) / len(y)
		### END YOUR CODE
        return score
    
    def assign_weights(self, weights):
        self.W = weights
        return self

