import math
from cProfile import label
import os
import matplotlib.pyplot as plt
from LogisticRegression import logistic_regression
from LRM import logistic_regression_multiclass
from DataReader import *

data_dir = "../data/"
train_filename = "training.npz"
test_filename = "test.npz"

# helper function for legend in plots
def legend_without_duplicate_labels(ax):
    handles, labels = ax.get_legend_handles_labels()
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
    ax.legend(*zip(*unique))
    
def visualize_features(X, y):
    '''This function is used to plot a 2-D scatter plot of training features. 

    Args:
        X: An array of shape [n_samples, 2].
        y: An array of shape [n_samples,]. Only contains 1 or -1.

    Returns:
        No return. Save the plot to 'train_features.*' and include it
        in submission.
    '''
    ### YOUR CODE HERE
    plt.clf()
    fig, ax = plt.subplots()
    ax.plot(X[np.where(y==1),0],X[np.where(y==1),1],c='green',marker='o',markersize=4,label='Class +1')
    ax.plot(X[np.where(y==-1),0],X[np.where(y==-1),1],c='blue',marker='x',markersize=4,label='Class -1')
    legend_without_duplicate_labels(ax)
    plt.xlabel('Feature 1 - Symmetry')
    plt.ylabel('Feature 2 - Intensity')
    plt.savefig('train_features.jpg')
    ### END YOUR CODE

def visualize_result(X, y, W):
    '''This function is used to plot the sigmoid model after training.

	Args:
		X: An array of shape [n_samples, 2].
		y: An array of shape [n_samples,]. Only contains 1 or -1.
		W: An array of shape [n_features,].
	
	Returns:
		No return. Save the plot to 'train_result_sigmoid.*' and include it
		in submission.
	'''
	### YOUR CODE HERE
    plt.clf()
    fig, ax = plt.subplots()
    ax.plot(X[np.where(y==1),0],X[np.where(y==1),1],c='green',marker='o',markersize=4,label='Class +1')
    ax.plot(X[np.where(y==-1),0],X[np.where(y==-1),1],c='blue',marker='x',markersize=4,label='Class -1')
    legend_without_duplicate_labels(ax)
    plt.xlabel('Feature 1 - Symmetry') # x1
    plt.ylabel('Feature 2 - Intensity') # x2
    # now we plot decision boundary
    # WT.x = 0 is the line equation where w =[w0,w1,w2] and x=[1,x1,x2]
    # W0+w1x1+W2x2 = 0 => x2 = -(W0+W1x1)/W2
    dummy_vals_along_x_axis = np.array([X[:,0].min(), X[:,0].max()])
    dummy_vals_along_y_axis = -(W[0]+W[1]*dummy_vals_along_x_axis)/W[2]
    plt.plot(dummy_vals_along_x_axis,dummy_vals_along_y_axis,c="#000",label="Decision boundary")
    plt.xlim([-1,0])
    plt.ylim([-1,0])
    plt.savefig('train_result_sigmoid.jpg')
    ### END YOUR CODE

def visualize_result_multi(X, y, W):
    '''This function is used to plot the softmax model after training. 

	Args:
		X: An array of shape [n_samples, 2].
		y: An array of shape [n_samples,]. Only contains 0,1,2.
		W: An array of shape [n_features, 3].
	
	Returns:
		No return. Save the plot to 'train_result_softmax.*' and include it
		in submission.
	'''
	### YOUR CODE HERE
    plt.clf()
    fig, ax = plt.subplots()
    ax.plot(X[np.where(y==0),0], X[np.where(y==0),1], c="red", marker="o", markersize=4, label="Class 0") # class 0
    ax.plot( X[np.where(y==1),0], X[np.where(y==1),1],c="green",marker="x", markersize=4, label="Class 1") # class 1
    ax.plot( X[np.where(y==2),0], X[np.where(y==2),1],c="blue",marker="d", markersize=4, label="Class 2") # class 2
    legend_without_duplicate_labels(ax)
    plt.xlabel('Feature 1 - Symmetry')
    plt.ylabel('Feature 2 - Intensity')
    dummy_vals_along_x_axis = np.linspace(X[:,0].min(), X[:,0].max(),50)

    # to compute the angular bisectors, we first normalize each weight vector (a,b,c) by dividing it with sqrt(a2+b2)
    W0,W1,W2 = W[:,0],W[:,1],W[:,2]
    normalize = lambda w: w/math.sqrt(w[1]**2+w[2]**2)
    W0,W1,W2 = normalize(W0),normalize(W1),normalize(W2)
    line_eq = lambda weight_vec: -(weight_vec[0]+weight_vec[1]*dummy_vals_along_x_axis)/weight_vec[2]

    dummy_vals_along_y_axis = np.maximum(line_eq(W2 - W0),line_eq(W0-W1))
    plt.plot(dummy_vals_along_x_axis, dummy_vals_along_y_axis, c="black", linestyle="dashed")
    dummy_vals_along_y_axis = np.minimum(line_eq(W1 - W2), line_eq(W0 - W1))
    plt.plot(dummy_vals_along_x_axis, dummy_vals_along_y_axis, c="black", linestyle="dashed")
    plt.xlim([-1,0])
    plt.ylim([-1,0])
    plt.savefig('train_result_softmax.jpg')

	### END YOUR CODE

def main():
	# ------------Data Preprocessing------------
	# Read data for training.
    
    raw_data, labels = load_data(os.path.join(data_dir, train_filename))
    raw_train, raw_valid, label_train, label_valid = train_valid_split(raw_data, labels, 2300)

    ##### Preprocess raw data to extract features
    train_X_all = prepare_X(raw_train)
    valid_X_all = prepare_X(raw_valid)
    ##### Preprocess labels for all data to 0,1,2 and return the idx for data from '1' and '2' class.
    train_y_all, train_idx = prepare_y(label_train)
    valid_y_all, val_idx = prepare_y(label_valid)  

    ####### For binary case, only use data from '1' and '2'  
    train_X = train_X_all[train_idx]
    train_y = train_y_all[train_idx]
    ####### Only use the first 1350 data examples for binary training. 
    train_X = train_X[0:1350]
    train_y = train_y[0:1350]
    valid_X = valid_X_all[val_idx]
    valid_y = valid_y_all[val_idx]
    ####### set lables to  1 and -1. Here convert label '2' to '-1' which means we treat data '1' as postitive class. 
    train_y[np.where(train_y==2)] = -1
    valid_y[np.where(valid_y==2)] = -1
    data_shape= train_y.shape[0] 

#    # Visualize training data.
    visualize_features(train_X[:, 1:3], train_y)


   # ------------Logistic Regression Sigmoid Case------------

   ##### Check BGD, SGD, miniBGD
    logisticR_classifier = logistic_regression(learning_rate=0.5, max_iter=100)

    logisticR_classifier.fit_BGD(train_X, train_y)
    print(logisticR_classifier.get_params())
    print(logisticR_classifier.score(train_X, train_y))

    logisticR_classifier.fit_miniBGD(train_X, train_y, data_shape)
    print(logisticR_classifier.get_params())
    print(logisticR_classifier.score(train_X, train_y))

    logisticR_classifier.fit_SGD(train_X, train_y)
    print(logisticR_classifier.get_params())
    print(logisticR_classifier.score(train_X, train_y))

    logisticR_classifier.fit_miniBGD(train_X, train_y, 1)
    print(logisticR_classifier.get_params())
    print(logisticR_classifier.score(train_X, train_y))

    logisticR_classifier.fit_miniBGD(train_X, train_y, 10)
    print(logisticR_classifier.get_params())
    print(logisticR_classifier.score(train_X, train_y))


    # Explore different hyper-parameters.
    ### YOUR CODE HERE
    # we have two hyper-parameters - learning rate and batch size
    best_logisticR_score = logisticR_classifier.score(valid_X, valid_y) # mini batch with batch size 10 gave the best model until now
    best_logisticR = logisticR_classifier
    lrs = [0.01,0.03,0.09,0.3,0.6,0.9] # learning rates
    for learning_rate in lrs:
        logisticR_classifier = logistic_regression(learning_rate=learning_rate,max_iter=500)
        print("Training miniBGD with learning_rate: ",learning_rate)
        logisticR_classifier.fit_miniBGD(train_X,train_y,batch_size=10)
        score = logisticR_classifier.score(valid_X,valid_y)
        print("validation acc: ",score)
        if best_logisticR_score<score:
            best_logisticR_score = score
            best_logisticR = logisticR_classifier
    ### END YOUR CODE

	# Visualize the your 'best' model after training.
	# visualize_result(train_X[:, 1:3], train_y, best_logisticR.get_params())

    ### YOUR CODE HERE
    visualize_result(train_X[:,1:3],train_y,best_logisticR.get_params())
    ### END YOUR CODE

    # Use the 'best' model above to do testing. Note that the test data should be loaded and processed in the same way as the training data.
    ### YOUR CODE HERE

    # prepare test data and labels
    raw_data_test, labels_test = load_data(os.path.join(data_dir, test_filename))
    test_X_all,(test_y_all,binary_idx) = prepare_X(raw_data_test),prepare_y(labels_test)
    test_y = test_y_all[binary_idx]
    test_X = test_X_all[binary_idx]
    ####### set lables to  1 and -1. Here convert label '2' to '-1' which means we treat data '1' as postitive class. 
    test_y[np.where(test_y==2)] = -1

    # run predictions on test data and get mean accuracy
    test_acc = best_logisticR.score(test_X,test_y)
    print("Test accuracy for the best logistic regression model is: ",test_acc)
    ### END YOUR CODE


    # ------------Logistic Regression Multiple-class case, let k= 3------------
    ###### Use all data from '0' '1' '2' for training
    train_X = train_X_all
    train_y = train_y_all
    valid_X = valid_X_all
    valid_y = valid_y_all

    #########  miniBGD for multiclass Logistic Regression
    logisticR_classifier_multiclass = logistic_regression_multiclass(learning_rate=0.5, max_iter=100,  k= 3)
    logisticR_classifier_multiclass.fit_miniBGD(train_X, train_y, 10)
    print(logisticR_classifier_multiclass.get_params())
    print(logisticR_classifier_multiclass.score(train_X, train_y))

    # Explore different hyper-parameters.
    ### YOUR CODE HERE
    best_logisticR_multiclass_score = logisticR_classifier_multiclass.score(valid_X, valid_y) # mini batch with batch size 10 gave the best model until now
    best_logistic_multi_R = logisticR_classifier_multiclass
    lrs = [0.01,0.03,0.09,0.3,0.6,0.9] # learning rates
    for learning_rate in lrs:
        logisticR_classifier_multiclass = logistic_regression_multiclass(learning_rate=learning_rate,max_iter=500, k=3)
        print("Training multiclass miniBGD with learning_rate: ",learning_rate)
        logisticR_classifier_multiclass.fit_miniBGD(train_X,train_y,batch_size=10)
        score = logisticR_classifier_multiclass.score(valid_X,valid_y)
        print("validation acc: ",score)
        if best_logisticR_multiclass_score<score:
            best_logisticR_multiclass_score = score
            best_logistic_multi_R = logisticR_classifier_multiclass
    ### END YOUR CODE

	# Visualize the your 'best' model after training.
	# visualize_result_multi(train_X[:, 1:3], train_y, best_logistic_multi_R.get_params())


    # Use the 'best' model above to do testing.
    ### YOUR CODE HERE
    visualize_result_multi(train_X[:, 1:3], train_y, best_logistic_multi_R.get_params())
    print("Test accuracy for the best multiclass model is: ",best_logistic_multi_R.score(test_X_all,test_y_all))
    ### END YOUR CODE


    # ------------Connection between sigmoid and softmax------------
    ############ Now set k=2, only use data from '1' and '2' 

    #####  set labels to 0,1 for softmax classifer
    train_X = train_X_all[train_idx]
    train_y = train_y_all[train_idx]
    train_X = train_X[0:1350]
    train_y = train_y[0:1350]
    valid_X = valid_X_all[val_idx]
    valid_y = valid_y_all[val_idx] 
    train_y[np.where(train_y==2)] = 0
    valid_y[np.where(valid_y==2)] = 0  
    
    ###### First, fit softmax classifer until convergence, and evaluate 
    ##### Hint: we suggest to set the convergence condition as "np.linalg.norm(gradients*1./batch_size) < 0.0005" or max_iter=10000:
    ### YOUR CODE HERE
    logisticR_classifier_multiclass = logistic_regression_multiclass(learning_rate=0.5,max_iter=10000,k=2)
    logisticR_classifier_multiclass.fit_miniBGD(train_X, train_y, batch_size=10)
    print("softmax classififer acc on train data with 2 classes =",logisticR_classifier_multiclass.score(train_X,train_y))
    print("softmax classififer acc on validation data with 2 classes =",logisticR_classifier_multiclass.score(valid_X,valid_y))
    test_X = test_X_all[binary_idx]
    test_y = test_y_all[binary_idx]
    test_y[np.where(test_y==2)] = 0
    print("softmax classififer acc on test data with 2 classes =", logisticR_classifier_multiclass.score(test_X, test_y))
    print("softmax classifier weights=",logisticR_classifier_multiclass.get_params())
    ### END YOUR CODE






    train_X = train_X_all[train_idx]
    train_y = train_y_all[train_idx]
    train_X = train_X[0:1350]
    train_y = train_y[0:1350]
    valid_X = valid_X_all[val_idx]
    valid_y = valid_y_all[val_idx] 
    #####       set lables to -1 and 1 for sigmoid classifer
    train_y[np.where(train_y==2)] = -1
    valid_y[np.where(valid_y==2)] = -1   

    ###### Next, fit sigmoid classifer until convergence, and evaluate
    ##### Hint: we suggest to set the convergence condition as "np.linalg.norm(gradients*1./batch_size) < 0.0005" or max_iter=10000:
    ### YOUR CODE HERE
    logisticR_classifier = logistic_regression(learning_rate=0.5,max_iter=10000)
    logisticR_classifier.fit_miniBGD(train_X,train_y,batch_size=10)
    print("logistic classififer acc on train data with 2 classes =",logisticR_classifier.score(train_X,train_y))
    print("logisitic classififer acc on validation data with 2 classes =",logisticR_classifier.score(valid_X,valid_y))
    test_X = test_X_all[binary_idx]
    test_y = test_y_all[binary_idx]
    test_y[np.where(test_y==2)] = -1
    print("logistic classififer acc on test data with 2 classes =", logisticR_classifier.score(test_X, test_y))
    print("logistic classifier weights=",logisticR_classifier.get_params())
    ### END YOUR CODE


    ################Compare and report the observations/prediction accuracy


    '''
    Explore the training of these two classifiers and monitor the graidents/weights for each step. 
    Hint: First, set two learning rates the same, check the graidents/weights for the first batch in the first epoch. What are the relationships between these two models? 
    Then, for what leaning rates, we can obtain w_1-w_2= w for all training steps so that these two models are equivalent for each training step. 
    '''
    ### YOUR CODE HERE

    # set learning rate
    learning_rate_softmax = 0.1

    # check logistic weight after one class

    logisticR_classifier = logistic_regression(learning_rate=2*learning_rate_softmax,max_iter=2)
    logisticR_classifier.fit_miniBGD_print_weights(train_X,train_y,batch_size=10)

    # check softmax weights after one class
    train_y[np.where(train_y==-1)] = 0 # set -1 class back to label 0

    logisticR_classifier_multiclass = logistic_regression_multiclass(learning_rate=learning_rate_softmax,max_iter=2,k=2)
    logisticR_classifier_multiclass.fit_miniBGD_print_weights(train_X,train_y,batch_size=10)
    

    ### END YOUR CODE

    # ------------End------------
    

if __name__ == '__main__':
	main()
    
    
