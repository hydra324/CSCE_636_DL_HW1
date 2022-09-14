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
    pos_class = np.where(y==1)
    neg_class = np.where(y==-1)
    pos_plt = plt.scatter(X[pos_class,0],X[pos_class,1],c='#1f77b4',marker='o')
    neg_plt = plt.scatter(X[neg_class,0],X[neg_class,1],c='#6f6f6f',marker='x')
    plt.legend((pos_plt,neg_plt),('Class +1','Class -1'))
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
    pos_class = np.where(y==1)
    neg_class = np.where(y==-1)
    pos_plt = plt.plot(X[pos_class,0],X[pos_class,1],c='#1f77b4',marker='o')
    neg_plt = plt.plot(X[neg_class,0],X[neg_class,1],c='#6f6f6f',marker='x')
    # plt.legend((pos_plt,neg_plt),('Class +1','Class -1'))
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
    plt.plot(X[np.where(y==0),0], X[np.where(y==0),1], c="red", marker="o", markersize=4) # class 0
    plt.plot( X[np.where(y==1),0], X[np.where(y==1),1],c="green",marker="x", markersize=4) # class 1
    plt.plot( X[np.where(y==2),0], X[np.where(y==2),1],c="blue",marker="d", markersize=4) # class 2
    plt.xlabel('Feature 1 - Symmetry')
    plt.ylabel('Feature 2 - Intensity')
    # now we plot all 3 decision boundaries
    # dummy_vals_along_x_axis = np.array([X[:,0].min(), X[:,0].max()])
    dummy_vals_along_x_axis = np.linspace(X[:,0].min(), X[:,0].max(),50)
    # x0,x1 = np.meshgrid(
    #     np.linspace(X[:,0].min(),X[:,0].max(),500).reshape(-1,-1),
    #     np.linspace(X[:,1].min(),X[:,1].max(),500).reshape(-1,1)
    # )

    W0,W1,W2 = W[:,0],W[:,1],W[:,2]
    normalize = lambda w: w/math.sqrt(w[1]**2+w[2]**2)
    W0,W1,W2 = normalize(W0),normalize(W1),normalize(W2)
    # W01 = W0-W1
    # W10 = W1-W0
    # W12 = W1-W2
    # W02 = W0-W2
    # # compare 01,02
    # dummy_vals_along_y_axis = np.max([-(W01[0]+W01[1]*dummy_vals_along_x_axis)/W01[2],-(W02[0]+W02[1]*dummy_vals_along_x_axis)/W02[2]],axis=0)
    # plt.plot(dummy_vals_along_x_axis, dummy_vals_along_y_axis, c="black", linestyle="dashed")
    #
    # # compare 10 and 12
    # dummy_vals_along_y_axis = np.max(
    #     [-(W10[0] + W10[1] * dummy_vals_along_x_axis) / W10[2], -(W12[0] + W12[1] * dummy_vals_along_x_axis) / W12[2]],axis=0)
    # plt.plot(dummy_vals_along_x_axis, dummy_vals_along_y_axis, c="black", linestyle="dashed")

    # decision boundary class 0
    # weight_vec = W[:,0]
    # weight_vec = W0-W1
    line_eq = lambda weight_vec: -(weight_vec[0]+weight_vec[1]*dummy_vals_along_x_axis)/weight_vec[2]
    # dummy_vals_along_y_axis = np.max([line_eq(W0-W1)],axis=0)
    # dummy_vals_along_y_axis = -(weight_vec[0]+weight_vec[1]*dummy_vals_along_x_axis)/weight_vec[2]
    # plt.plot(dummy_vals_along_x_axis,dummy_vals_along_y_axis,c="red",linestyle="dashed")

    # # decision boundary class 1
    # # weight_vec = W[:,1]
    # weight_vec = W1-W2
    # # dummy_vals_along_y_axis = np.minimum(line_eq(W0 - W1), line_eq(W1 - W2))
    # dummy_vals_along_y_axis = -(weight_vec[0]+weight_vec[1]*dummy_vals_along_x_axis)/weight_vec[2]
    # dummy_vals_along_y_axis = line_eq(W1-W2)
    # plt.plot(dummy_vals_along_x_axis,dummy_vals_along_y_axis,c="green",linestyle="dashed")
    # #
    # # decision boundary class 2
    # # weight_vec = W[:,2]
    # weight_vec = W2-W0
    # dummy_vals_along_y_axis = -(weight_vec[0]+weight_vec[1]*dummy_vals_along_x_axis)/weight_vec[2]
    # dummy_vals_along_y_axis = line_eq(W2-W0)
    # plt.plot(dummy_vals_along_x_axis,dummy_vals_along_y_axis,c="blue",linestyle="dashed")

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
    best_batch_size = 10
    # # lrs = [0.01,0.03,0.09,0.3,0.6,0.9] # learning rates
    # # batch_sizes = [1,10,100,500,data_shape]
    # for learning_rate in lrs:
    #     for batch_size in batch_sizes:
    #         logisticR_classifier = logistic_regression(learning_rate=learning_rate,max_iter=500)
    #         print("Training miniBGD with batch_size: ",batch_size, "and learning_rate: ",learning_rate)
    #         logisticR_classifier.fit_miniBGD(train_X,train_y,batch_size=batch_size)
    #         score = logisticR_classifier.score(valid_X,valid_y)
    #         if best_logisticR_score<score:
    #             best_logisticR_score = score
    #             best_logisticR = logisticR_classifier
    #             best_batch_size = batch_size

    # re-train the best model on initial tranining set i.e, train+validation
    # the reason we aren't using train_y_all here is because the labels for class 2 arent replaced with -1 in it.
    # train_and_valid_X = np.concatenate((train_X,valid_X))
    # train_and_valid_y = np.concatenate((train_y,valid_y))
    # best_logisticR.fit_miniBGD(train_and_valid_X,train_and_valid_y,best_batch_size)
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
    print("Test accuracy for the best model is: ",test_acc)
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
    logisticR_classifier_multiclass = logistic_regression_multiclass(learning_rate=0.1, max_iter=1000,  k= 3)
    logisticR_classifier_multiclass.fit_miniBGD(train_X, train_y, 10)
    print(logisticR_classifier_multiclass.score(valid_X, valid_y))
    ### END YOUR CODE

	# Visualize the your 'best' model after training.
	# visualize_result_multi(train_X[:, 1:3], train_y, best_logistic_multi_R.get_params())


    # Use the 'best' model above to do testing.
    ### YOUR CODE HERE
    best_logistic_multi_R = logisticR_classifier_multiclass
    visualize_result_multi(train_X[:, 1:3], train_y, best_logistic_multi_R.get_params())
    print("Test accuracy for the best model is: ",best_logistic_multi_R.score(test_X_all,test_y_all))
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
    # logisticR_classifier_multiclass = logistic_regression_multiclass(learning_rate=0.5,max_iter=10000,k=2)
    # logisticR_classifier_multiclass.fit_miniBGD(train_X, train_y, 10)
    

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

    ### END YOUR CODE


    ################Compare and report the observations/prediction accuracy


'''
Explore the training of these two classifiers and monitor the graidents/weights for each step. 
Hint: First, set two learning rates the same, check the graidents/weights for the first batch in the first epoch. What are the relationships between these two models? 
Then, for what leaning rates, we can obtain w_1-w_2= w for all training steps so that these two models are equivalent for each training step. 
'''
    ### YOUR CODE HERE

    ### END YOUR CODE

    # ------------End------------
    

if __name__ == '__main__':
	main()
    
    
