# Program structure referred from http://machinelearningmastery.com/tutorial-to-implement-k-nearest-neighbors-in-python-from-scratch/

import numpy as np
import scipy.io as sio
import operator
from scipy.spatial.distance import cosine #to compute cosine distance
import matplotlib.pyplot as plt




# loading the data from the matlab file and converting them to numpy array
mat_contents = sio.loadmat('faces.mat')
train_labels = np.array(mat_contents['trainlabels'])
train_data = np.array(mat_contents['traindata'])
test_data = np.array(mat_contents['testdata'])
test_labels = np.array(mat_contents['testlabels'])
eval_data = np.array(mat_contents['evaldata'])
full_train_data = np.hstack((train_data, train_labels)) # combine label as the last column
full_test_data = np.hstack((test_data,test_labels))


def get_neighbors(train_egs,test_case,k): # get nearest neighbors
    i=0
    dist = []
    for i in range(len(train_egs)):
        temp_dist = cosine(test_case,train_egs[i])
        dist.append((train_egs[i],temp_dist)) # dist = [(training example, cosine dist)]
    dist.sort(key=operator.itemgetter(1)) # sort distance based on cosine distances

    neighbor_list = []
    i=0
    for i in range(k):
        neighbor_list.append(dist[i][0]) # list of k nearest neighbors in ascending cosine distance
    return neighbor_list

def get_votes(neighbors): # weight of nearest neighbors, Parameter neighbor = neighbor_list
    votes={}
    for i in range(len(neighbors)):
        response = neighbors[i][-1] # assign response as the label of the neighbor
        if response in votes: # check if response already in votes, if its there increment count by 1
            votes[response] += 1
        else:   # if response not here, then assign it value 1
            votes[response] = 1
    sort_votes = sorted(votes.items(), key=operator.itemgetter(1), reverse = True) # sort the votes in ascending order
    return sort_votes[0][0] # return the  nearest neighbor with the highest vote count

def get_correct(test_egs, predictions):  # get the % accuracy of our predictions
    correct = 0
    for i in range(len(test_egs)):
        if test_egs[i][-1] == predictions[i]:
            correct +=1

    percentage = (correct/float(len(test_egs)))*100
    return percentage


error_test = []
error_train = []
k_values = [1,10,20,30,40,50,60,70,80,90,100]  # list with the given k values
print ('---Starting on test data set---')
for k in k_values:  # for testing data error
    print ('K=',k)
    predictions = []
    for i in range(len(full_test_data)):

        near_neighbor = get_neighbors(full_train_data,full_test_data[i],k)
        votes = get_votes(near_neighbor)
        predictions.append((votes)) # make array predictions which has all the predicted values
    print ('Predictions---',predictions)
    accuracy = get_correct(full_test_data,predictions)
    error = 100 - accuracy
    print ('Error-',error,'Accuracy-',accuracy)
    error_test.append((error))
print ('Test Errors-',error_test)
print ('---Starting on training data set---')


k_values = [1,10,20,30,40,50,60,70,80,90,100]
for k in k_values:      # for training data error
    print ('K=',k)
    predictions = []
    for i in range(len(full_train_data)):

        near_neighbor = get_neighbors(full_train_data,full_train_data[i],k)
        votes = get_votes(near_neighbor)
        predictions.append((votes))
    print ('Predictions---',predictions)
    accuracy = get_correct(full_train_data,predictions)
    error = 100 - accuracy
    print ('Error-',error,'Accuracy-',accuracy)
    error_train.append((error))
print('Training errors-',error_train)
print ('Test Errors-',error_test)



line_up, = plt.plot(k_values,error_test)
line_down, = plt.plot(k_values,error_train)
plt.ylabel('Error')
plt.xlabel('K Value')
plt.legend([line_up, line_down], ['Test Data', 'Training Data']) # plotting the training error and the test error
plt.show()
