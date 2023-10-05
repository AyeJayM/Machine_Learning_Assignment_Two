#-------------------------------------------------------------------------
# AUTHOR: AUSTIN MARTINEZ
# FILENAME: tnaive_bayes.py
# SPECIFICATION: We read in a CSV file and use a dictionary to assign weather conditions a numerical value. This will allow us to train
# our model using the Naive Bayesian method. The program will generate two confidence values when we use our trained model to predict 
# whether the weather is suitable for playing tennis. The higher confidence value will determine if we select "Yes" or "No."
# FOR: CS 4210- Assignment #2
# TIME SPENT: 12 hours
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard
# dictionaries, lists, and arrays

#importing some Python libraries
from sklearn.naive_bayes import GaussianNB
import csv

#reading the training data in a csv file
#--> add your Python code here
db = []
X = []
Y = []

#reading the data in a csv file
with open('weather_training.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    for i, row in enumerate(reader):
      if i > 0: #skipping the header
         db.append (row)

#transform the original training features to numbers and add them to the 4D array X.
#For instance Sunny = 1, Overcast = 2, Rain = 3, so X = [[3, 1, 1, 2], [1, 3, 2, 2], ...]]
#--> add your Python code here
# X =

# Assign each feature value a numerical value
featuresDict = {
    #Outlook
    "Sunny" : 1,
    "Overcast" : 2,
    "Rain" : 3,

    #Temperature
    "Hot" : 1,
    "Mild" : 2,
    "Cool" : 3,

    #Humidity
    "Normal" : 1,
    "High" : 2,

    #Wind
    "Weak" : 1,
    "Strong" : 2
}

# Set our row and column values equal to the dimensions of db matrix
rows = len(db)
cols = len(db[0])-2 # "-1" because we don't want to include the class label or day in our feature values matrix

# Initialize feature values matrix size
for i in range(rows):
    row = [0] * cols
    X.append(row)

# Convert feature values using a dictionary and assign them to feature values matrix
    # We print the matrix to display the feature values converted into numerical values
print(f"\n--- FEATURE VALUES MATRIX ---")

for row in range(rows):
    output = ""
    for col in range(cols):
        X[row][col] = featuresDict.get(db[row][col+1], db[row][col+1]) # Recall "+1" because we skip the day value
        output += str(X[row][col]) + " "
    print(output)


#transform the original training classes to numbers and add them to the vector Y.
#For instance Yes = 1, No = 2, so Y = [1, 1, 2, 2, ...]
#--> add your Python code here
# Y =

# Assign each class value a numerical value
classDict = {
    #Reccomended Lenses
    "No" : 1,
    "Yes" : 2
}

#Initialize class values matrix size
for row in range(rows):
    row = [0] * 1
    Y.append(row)

# Convert class values using a dictionary and assign them to class values matrix
    # We print the matrix to display the class values converted into numerical values
print(f"\n--- CLASS LABELS VECTOR ---")
for row in range(rows):
    output = ""
    Y[row][0] = classDict.get(db[row][-1], db[row][-1]) # "-1" to acess the last column in the row
    output += str(Y[row][0]) + " "

# Convert Y to a horizontal vector using list comprehension so it will be accepted by the Guassian function
Y = [item[0] for item in Y]
print(Y)

#fitting the naive bayes to the data
clf = GaussianNB()
clf.fit(X, Y)

#reading the test data in a csv file
#--> add your Python code here
testDB = []
testX = []
with open('weather_test.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    for i, row in enumerate(reader):
      if i > 0: #skipping the header
         testDB.append (row)

# Set our row and column values equal to the dimensions of db matrix
testRows = len(testDB)
testCols = len(testDB[0])-2 # "-1" because we don't want to include the class label or day in our feature values matrix

# Initialize feature values matrix size
for i in range(testRows):
    newRow = [0] * testCols
    testX.append(newRow)

print(f"\n--- TEST DATA FEATURE VALUES MATRIX ---")

for row in range(testRows):
    output = ""
    for col in range(testCols):
        testX[row][col] = featuresDict.get(testDB[row][col+1], testDB[row][col+1]) # Recall "+1" because we skip the day value
        output += str(testX[row][col]) + " "
    print(output)

#printing the header of the solution
#--> add your Python code here
print(f"\n\nDay\t\tOutlook\t\tTemperature\t\tHumidity\tWind\t\t\tPlayTennis\t\tConfidence")

#use your test samples to make probabilistic predictions. For instance: clf.predict_proba([[3, 1, 2, 1]])[0]
#--> add your Python code here

for row in range(testRows):
    probabilities = clf.predict_proba([testX[row]])[0] # Do not use double brackets, otherwise we are passing in a 3D array.

    if probabilities[0] >= 0.75:
        output = ""
        for col in range(testCols+1):
                output += str(testDB[row][col])
                output += "\t\t"
        output += "\tNo\t\t"
        output += str(probabilities[0])
        print(output)

    if probabilities[1] >= 0.75:
        output = ""
        for col in range(testCols+1):
                output += str(testDB[row][col])
                output += "\t\t"
        output += "\t\tYes\t\t"
        output += str(probabilities[1])
        print(output)



