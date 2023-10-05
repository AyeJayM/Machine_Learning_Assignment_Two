#-------------------------------------------------------------------------
# AUTHOR: AUSTIN MARTINEZ
# FILENAME: knn.py
# SPECIFICATION: Read in a CSV file and assign the value pair to an array and the class label to a vector. We then train our
# model using the principles of leave-one-out-cross-validation error rate to calculate an error rate for 1NN. We try and predict
# an instance using the nearest neighbor.
# FOR: CS 4210- Assignment #2
# TIME SPENT: 12 hours.
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard vectors and arrays

#importing some Python libraries
from sklearn.neighbors import KNeighborsClassifier
import csv

db = []

#reading the data in a csv file
with open('binary_points.csv', 'r') as csvfile:
  reader = csv.reader(csvfile)
  for i, row in enumerate(reader):
      if i > 0: #skipping the header
         db.append (row)

# ---------------------------
#
# Retrieve the pairs for X
#
# ---------------------------

X = []
for row in db:
    # Extract the first two values, convert them to floats, and store them as a pair
    x, y = map(float, row[:2])
    X.append([x, y])

#print(X)

# ------------------------------
#
# Retrieve the class labels for Y
#
# ------------------------------

Y = []
labelsDict = {'-': 1.0, '+': 2.0}
for row in db:
    # Extract the last value, convert it to a float, and store them
    cur_value = row[-1]
    Y.append(labelsDict[cur_value])

#print(Y)

curInstance = 0
totalPredictions = 0
incorrectPredictions = 0

#loop your data to allow each instance to be your test set
for item in db:

    #add the training features to the 2D array X removing the instance that will be used for testing in this iteration. For instance, X = [[1, 3], [2, 1,], ...]]. Convert each feature value to
    # float to avoid warning messages
    #--> add your Python code here
    # X = 

    newX = [] # This will be the 2D array we use for testing
    for i in range(len(X)): # x - 1 because we are using one instance as testing
        if i != curInstance: # Copy to a new list and skip over the instance used for testing
            newX.append(X[i]) # Append any other values to the newX list

    print(f"\n\n\n\n\nCurrent X 2D-array for {curInstance+1} is:\n{newX}")
    print("\n")

    #transform the original training classes to numbers and add to the vector Y removing the instance that will be used for testing in this iteration. For instance, Y = [1, 2, ,...]. Convert each
    #  feature value to float to avoid warning messages
    #--> add your Python code here
    # Y =

    newY = [] # This will be the 2D array we use for testing
    for i in range(len(X)): # x - 1 because we are using one instance as testing
        if i != curInstance: # Copy to a new list and skip over the instance used for testing
            newY.append(Y[i]) # Append any other values to the newX list

    print(f"Current Y 2D-array for {curInstance+1} is:\n{newY}")
    print("\n")

    #store the test sample of this iteration in the vector testSample
    #--> add your Python code here
    #testSample =
    
    testSample = []
    testSample.append(X[curInstance])
    testSample.append(Y[curInstance])

    #fitting the knn to the data
    clf = KNeighborsClassifier(n_neighbors=1, p=2)
    clf = clf.fit(newX, newY)

    #use your test sample in this iteration to make the class prediction. For instance:
    #class_predicted = clf.predict([[1, 2]])[0]
    #--> add your Python code here

    class_predicted = clf.predict( [testSample[0]] )[0]

    print(f"Predicted Class is: {class_predicted}")
    print(f"Test Sample Class is: {testSample[1]}")

    #compare the prediction with the true label of the test instance to start calculating the error rate.
    #--> add your Python code here

    if class_predicted != testSample[1]:
        incorrectPredictions += 1

    
    totalPredictions += 1
    curInstance += 1

#print the error rate
#--> add your Python code here
print(f"\n\n\n\nIncorrect Predictions: {incorrectPredictions}")
print(f"Total Predictions: {totalPredictions}")


errorRate = incorrectPredictions/totalPredictions
print(f"Error Rate: {errorRate}\n")






