#Kamaal Palmer
#To run this code, simply run it in python without any arguments
#It will create the DT.png and ANN.png (confidence interval pictures)

#Imports:
import numpy as np #to use numpy
import pandas as pd #To use Pandas
import matplotlib.pyplot as plt # To use MatPlotLib
import scipy.stats as stats #to use stats
import math #math import
from sklearn import tree #to use trees
from sklearn.metrics import confusion_matrix #to do confusion matrix
from sklearn.metrics import accuracy_score #to do accuracy score
from sklearn.model_selection import train_test_split #to do train tes$
from sklearn.model_selection import cross_val_score #to do cross vali$
from sklearn.neural_network import MLPClassifier #to do ANN
from sklearn.preprocessing import StandardScaler 

def main():
    file_name = "p2data/dow_jones_index.data.csv"
    #Set Column labels
    cols = ['quarter', 'open', 'high', 'low', 'close', 'volume',
            'percent_change_price', 'percent_change_volume_over_last_wk', 'previous_weeks_volume',
            'next_weeks_open', 'next_weeks_close', 'days_to_next_dividend',
            'percent_return_next_dividend', 'percent_change_next_weeks_price']
    #Read in Dataset        
    dataset = pd.read_csv(file_name, names=cols)
    #Set new column, make 2 bins
    newcol = pd.qcut(dataset['percent_change_next_weeks_price'], 2, labels=False)
    print("Print New Column:" , newcol)
    
    dataset['q_percent_change_next_weeks_price'] = newcol
    print("Print Dataset:" , dataset)

    # note: this is also a type cast from pandas datafram to numpy array
    dataset = dataset.as_matrix()
    #Slice dataset
    X = dataset[:,0:13]
    Y = dataset[:,14]
    print("xshape is: ", X.shape)
    print("yshape is :", Y.shape)
    
    print("X:" , X)
    print("Y:" , Y)
    
    #DT Section:
    DTscoreList = list()
    print("Default Decision tree:")
    runDecisionTree(X, Y, 9999, 2, 1, DTscoreList) #Default Decision Tree 
    print("Decision Tree Variation 1:")
    runDecisionTree(X, Y, 9999, 6, 2, DTscoreList) #Variation 1
    print("Decision Tree Variation 2:")
    runDecisionTree(X, Y, 3, 10, 10, DTscoreList)  #Variation 2
    print("Decision Tree Variation 3:")
    runDecisionTree(X, Y, 50, 6, 2, DTscoreList)   #Variation 3
    print("Decision Tree Variation 4:")
    runDecisionTree(X, Y, 10, 2, 1, DTscoreList)   #Variation 4
    print("Decision Tree Variation 5:")
    runDecisionTree(X, Y, 10, 8, 4, DTscoreList)   #Variation 5
    calcPValues(DTscoreList)
    calcConfidenceInterval(DTscoreList, "DT.png")
        
    #ANN Section:
    ANNscoreList = list()
    print("Default Neural network:")
    runANN(X, Y, 100, 1, 200, ANNscoreList)  #Default Neural Network
    print("Neural network variation 1:")
    runANN(X, Y, 25, 30, 10, ANNscoreList)   #Variation1
    print("Neural network variation 2:")
    runANN(X, Y, 15, 20,10000, ANNscoreList) #Variation 2
    print("Neural network variation 3:")
    runANN(X, Y, 20, 20,2000, ANNscoreList)  #Variation 3
    print("Neural network variation 4:")
    runANN(X, Y, 20, 30,2000, ANNscoreList)  #Variation 4
    print("Neural network variation 5:")
    runANN(X, Y, 25, 30,2000, ANNscoreList)  #Variation 5
    calcPValues(ANNscoreList)
    calcConfidenceInterval(ANNscoreList, "ANN.png")
    
#This function calculates the P Values of the scoresList it is given
#Simply give it a list of cross val scores and it will print out the
#P values for that list.
def calcPValues(scoresList):
    pair01 = stats.ttest_rel(scoresList[0], scoresList[1])
    print("The p-value for 0,1 is", pair01[1])
    pair02 = stats.ttest_rel(scoresList[1], scoresList[2])
    print("The p-value for 1,2 is", pair02[1])
    pair03 = stats.ttest_rel(scoresList[2], scoresList[3])
    print("The p-value for 2,3 is", pair03[1])
    pair04 = stats.ttest_rel(scoresList[3], scoresList[4])
    print("The p-value for 3,4 is", pair04[1])
    pair05 = stats.ttest_rel(scoresList[0], scoresList[2])
    print("The p-value for 0,2 is", pair05[1])
    pair06 = stats.ttest_rel(scoresList[0], scoresList[3])
    print("The p-value for 0,3 is", pair06[1])
    pair07 = stats.ttest_rel(scoresList[0], scoresList[4])
    print("The p-value for 0,4 is", pair07[1])
    pair08 = stats.ttest_rel(scoresList[1], scoresList[3])
    print("The p-value for 1,3 is", pair08[1])
    pair09 = stats.ttest_rel(scoresList[1], scoresList[4])
    print("The p-value for 1,4 is", pair09[1])
    pair10 = stats.ttest_rel(scoresList[2], scoresList[4])
    print("The p-value for 2,4 is", pair10[1])
                                                                                
#This section calculates the confusion matrix given the dataset
#It will use the given model, and the parameters are the X and y
#of the dataset. It will print out the confusion matrix
def confusionMatrix(X, y, model):
    #split up the data
    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y,
                                               test_size = .2,
                                               random_state=10)
    #fit the data to the model
    model.fit(Xtrain, ytrain)
    #predict 
    y_model = model.predict(Xtest)
    #Check confusion matrix here
    mat = confusion_matrix(ytest, y_model)
    print(mat)
    
#This function creates a decision tree model using the given parameters. 
#The parameters are the X and Y of the model, the max depth of the tree, 
#the min samples split of the tree, the minimum samples per leaf of the tree, 
#and the scores list. There is no return value, it only appends the cross val
#scores to the scores list
def runDecisionTree(X, y, maxDepth, minSamplesSplit, minSamplesLeaf, scoresList):
    #Create DT    
    dt = tree.DecisionTreeClassifier(criterion='entropy', max_depth = maxDepth,
                                     min_samples_split = minSamplesSplit,
                                     min_samples_leaf = minSamplesLeaf)
    #Create Model
    model = dt.fit(X,y)
    confusionMatrix(X, y, model)
    #Calculate Cross Val score
    scores = cross_val_score(dt, X, y, cv = 5)
    print(scores)
    scoresList.append(scores)
    
#This function calculates the confidence intervals using the given parameters
#The parameters are the cross validation scores list from the different variations (someList)
#and the file name to be saved. There is no return value    
def calcConfidenceInterval(someList, fileName):
    z_critical = stats.norm.ppf(q = 0.975) #given to us from lab
    intervals = list()
    #Calculate the means for each data part
    sample_mean = list()
    for elements in someList:
        sample_mean.append(np.mean(elements))
    print("Sample Means:")
    print(sample_mean)
    #Standard Deviation
    i= -1
    for sample in someList:
        i += 1
        #Get the mean from the list of means
        theMean = sample_mean[i]
        #calculate standard deviation
        stdDev = np.std(sample)
        #calculate error margin
        errorMargin = (z_critical*stdDev)/math.sqrt(5)
        intervals.insert(i, [theMean-errorMargin,theMean+errorMargin])
    print("Intervals are: ", intervals)
    #Set the figure
    fig = plt.figure(figsize=(9,9))
    labels = ["v1" , "v2", "v3", "v4", "v5", "v6"]
    xvals = np.arange(5, 35, 5)
    yerrors = [(top-bot)/2 for top,bot in intervals]
    plt.axis(xmin=0, xmax=35) #Sets range on the x-axis
    plt.xticks(xvals, labels)
    plt.errorbar(x = xvals,
                 y = sample_mean,
                 yerr=yerrors,
                 fmt='D')
    #show the figure
    #plt.show()
    
    #save the figure
    fig.savefig(fileName)
    
    
#This function creates a neural networks model using the given parameters. 
#The parameters	are the	X and Y	of the model, the number of nodes in the classifier,    
#the number of layers, the max number of iterations, 
#and the scores	list. There is no return value,	it only appends	the cross val
#scores	to the scores list
def runANN(X, y, numNodes, numLayers, numIterations, scoresList):
    #Create Nerual Network
    model = MLPClassifier(hidden_layer_sizes=numNodes*numLayers,
                          activation='logistic',
                          max_iter = numIterations,
                          random_state=0)
    confusionMatrix(X, y, model)
    #Scale the data
    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y,
                                                    test_size=.2,
                                                    random_state=0)
    scaler = StandardScaler()
    scaler.fit(Xtrain)
    
    Xtrain = scaler.transform(Xtrain)
    Xtest = scaler.transform(Xtest)
    
    X = scaler.transform(X)
    
    print('Number of samples in training set: %d' %len(ytrain))
    print('Number of samples in test set: %d' %len(ytest))
    
    model.fit(Xtrain,ytrain)
    #Calculate cross val score
    crossScore = cross_val_score(model, X, y, cv = 5)
    print(crossScore)
    scoresList.append(crossScore)
    
main()
