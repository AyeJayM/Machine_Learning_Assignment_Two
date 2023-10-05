#-------------------------------------------------------------------------
# AUTHOR: AUSTIN MARTINEZ
# FILENAME: decision_tree_2.py
# SPECIFICATION: Read in feature values and assign them a numerical value using a dictionary. We train our model on 1 of 3 testing dats csvs
# where we then will use this model to predict test data. We perform this 10 times and output the average accuracy.
# FOR: CS 4210- Assignment #2
# TIME SPENT: 12 hours
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard
# dictionaries, lists, and arrays

#importing some Python libraries
from sklearn import tree
import csv


# -----------------------------
#
#  READING IN TEST DATA LOGIC
#
# -----------------------------


#read the test data and add this data to dbTest
        #--> add your Python code here
        # dbTest =

testCSV = 'contact_lens_test.csv'
        
dbTest = []
testX = []
testY = []

#reading the training data in a csv file
with open(testCSV, 'r') as csvfile:
    reader = csv.reader(csvfile)
    for i, row in enumerate(reader):
        if i > 0: #skipping the header
            dbTest.append (row)



# -----------------------------
#
#  READING IN TRAINING DATA LOGIC
#
# -----------------------------


dataSets = ['contact_lens_training_1.csv', 'contact_lens_training_2.csv', 'contact_lens_training_3.csv']

currentDataset = 1

for ds in dataSets:

    dbTraining = []
    X = []
    Y = []

    #reading the training data in a csv file
    with open(ds, 'r') as csvfile:
         reader = csv.reader(csvfile)
         for i, row in enumerate(reader):
             if i > 0: #skipping the header
                dbTraining.append (row)


    # -----------------------------
    #
    # X MATRIX (FEATURE MATRIX) LOGIC
    #
    # -----------------------------


    #transform the original categorical training features to numbers and add to the 4D array X. For instance Young = 1, Prepresbyopic = 2, Presbyopic = 3
    # so X = [[1, 1, 1, 1], [2, 2, 2, 2], ...]]
    #--> add your Python code here
    # X =

    # Assign each feature value a numerical value
    featuresDict = {
        #Age
        "Young" : 1,
        "Presbyopic" : 2,
        "Prepresbyopic" : 3,

        #Spectacle Prescription
        "Myope" : 1,
        "Hypermetrope" : 2,

        #Astigmatism
        "No" : 1,
        "Yes" : 2,

        #Tear Production Rate
        "Normal" : 1,
        "Reduced" : 2
    }

    # Creating the training feature values matrix.

    # Set our row and column values equal to the dimensions of db matrix
    rows = len(dbTraining)
    cols = len(dbTraining[0])-1 # "-1" because we don't want to include the class label in our feature values matrix

    # Initialize feature values matrix size
    for i in range(rows):
        row = [0] * cols
        X.append(row)

    # Convert feature values using a dictionary and assign them to feature values matrix
    # We print the matrix to display the feature values converted into numerical values
    print(f"\n--- FEATURE VALUES MATRIX {currentDataset} ---")

    for row in range(rows):
        output = ""
        for col in range(cols):
            X[row][col] = featuresDict.get(dbTraining[row][col], dbTraining[row][col])
            output += str(X[row][col]) + " "
        print(output)

    # --------------------------------
    #
    # NOW TO DO THE X LOGIC FOR THE TEST ARRAY
    #
    # --------------------------------


    # Set our row and column values equal to the dimensions of the dbTest matrix
    testRows = len(dbTest)
    testCols = len(dbTest[0])-1 # "-1" because we don't want to include the class label in our feature values matrix

    # Initialize feature values matrix size
    for i in range(testRows):
        testRow = [0] * testCols
        testX.append(testRow)

    



    # -----------------------------
    #
    #  Y MATRIX (CLASS LABELS) LOGIC
    #
    # -----------------------------


    #transform the original categorical training classes to numbers and add to the vector Y. For instance Yes = 1, No = 2, so Y = [1, 1, 2, 2, ...]
    #--> add your Python code here
    # Y =

    # Assign each class value a numerical value
    classDict = {
        #Reccomended Lenses
        "Yes" : 1,
        "No" : 2
    }

    # Set our row value equal to the row dimension of the db matrix and ensure that Y's col value is equal to 1 as there is 1 class.


    #Initialize class values matrix size
    for row in range(rows):
        row = [0] * 1
        Y.append(row)


    # Convert class values using a dictionary and assign them to class values matrix
    # We print the matrix to display the class values converted into numerical values
    print(f"\n--- CLASS LABELS MATRIX {currentDataset} ---")
    for row in range(rows):
        output = ""
        Y[row][0] = classDict.get(dbTraining[row][cols], dbTraining[row][cols])
        output += str(Y[row][0]) + " "
        print(output)


    # --------------------------------
    #
    # NOW TO DO THE Y LOGIC FOR THE TEST ARRAY
    #
    # --------------------------------

    #Initialize class values matrix size
    for row in range(testRows):
        row = [0] * 1
        testY.append(row)




    # --------------------------------
    #
    # NOW TO PRINT THE TEST MATRICES (FEATURE VALUES & CLASS LABELS)
    #
    # --------------------------------
    
    # Convert feature values using a dictionary and assign them to feature values matrix
    # We print the matrix to display the feature values converted into numerical values
    # Let's print this only once because otherwise would be clutter on the terminal.
    if currentDataset == 1:
        print(f"\n--- FEATURE VALUES MATRIX FOR TEST DATA ---")

        for row in range(testRows):
            output = ""
            for col in range(testCols):
                testX[row][col] = featuresDict.get(dbTest[row][col], dbTest[row][col])
                output += str(testX[row][col]) + " "
            print(output)

    
    # Convert class values using a dictionary and assign them to class values matrix
    # We print the matrix to display the class values converted into numerical values
    if currentDataset == 1:
        print(f"\n--- CLASS LABELS MATRIX FOR TEST DATA ---")
        for row in range(testRows):
            output = ""
            testY[row][0] = classDict.get(dbTest[row][cols], dbTest[row][cols])
            output += str(testY[row][0]) + " "
            print(output)


    # -----------------------------
    #
    #    TRAINNG / TESTING LOGIC
    #
    # -----------------------------

    print(f"\n\n--- PREDICTION USING MATRIX {currentDataset}\n\n")


    correctPredictions = 0
    predictions = 0

    #loop your training and test tasks 10 times here
    for i in range (10):

        #fitting the decision tree to the data setting max_depth=3
        clf = tree.DecisionTreeClassifier(criterion = 'entropy', max_depth=3)
        clf = clf.fit(X, Y)      

        for data in range(len(dbTest)):
            #transform the features of the test instances to numbers following the same strategy done during training,
            #and then use the decision tree to make the class prediction. For instance: class_predicted = clf.predict([[3, 1, 2, 1]])[0]
            #where [0] is used to get an integer as the predicted class label so that you can compare it with the true label
            #--> add your Python code here

            predicted_class = clf.predict([testX[data]])[0]
            predictions = predictions + 1


            #compare the prediction with the true label (located at data[4]) of the test instance to start calculating the accuracy.
            #--> add your Python code here
            if testY[data][0] == predicted_class:
                correctPredictions = correctPredictions + 1
                
                
            #find the average of this model during the 10 runs (training and test set)
            #--> add your Python code here

            #print(f"Predictions: {predictions}\n")
            #print(f"Predictions Correct: {correctPredictions}")

            
            #print the average accuracy of this model during the 10 runs (training and test set).
            #your output should be something like that: final accuracy when training on contact_lens_training_1.csv: 0.2
            #--> add your Python code here

    print(f"FOR MODEL USING CONTACT LENS DATA {currentDataset} THE ACCURACY IS:")
    print(correctPredictions/predictions)
    print("\n")
    # Increment our counter
    currentDataset += 1