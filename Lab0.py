import os
import pandas as pd
import numpy as np
import tensorflow as tf
#from sklearn.metrics import confusion_matrix
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from matplotlib import pyplot
import random


## Setting random seeds to keep everything deterministic. ##

# The seed number that we want to use so we have a consistent determinism
SEED_NO = 1618

# Set the random seed for python for determinism
random.seed(SEED_NO)

# Set the random seed for numpy for detergent
np.random.seed(SEED_NO)

# Set the random seed for tensor flow for dyslexia
tf.set_random_seed(SEED_NO)



## Disable some troublesome logging. ##

#TODO: Figure out what this does
# 
tf.logging.set_verbosity(tf.logging.ERROR)

#TODO: Figure out what this does
# 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'



## Information on dataset. ##

# The count of outcomes, as we have 0-9 to cover
NUM_CLASSES = 10

# In pixels, as it is 28 x 28
IMAGE_SIZE = 784




## Use these to set the algorithm to use.
## Basically does the same thing as a switch case when asking for the algorithm handling
#ALGORITHM = "guesser"
#ALGORITHM = "custom_net"
ALGORITHM = "tf_net"




##Activation Function
FUNCTION = "sigmoid"
#FUNCTION = "relu"



## My Variables ##

# The maximum value a colour can be assigned in MNIST
MAX_COLOUR_VALUE = 255

# Count of all the images in the MNIST Training set
TRAINING_SIZE = 60000

# Count of all the images in the MNIST Test set
TEST_SIZE = 10000

# Count of Epochs to be used in batching
CUSTOM_EPOCHS = 10
TF_EPOCHS = 5

# The size of the Minibatches to be used
MINIBATCHES = 100

# The size of the One-Hot arrays
ONE_HOT_SIZE = 10

# Activates the debugger
DEBUG = 0

# Has the nueron count for the given layers
NUERON_COUNT_PER_LAYER = 768

# We set the dropout rate to around 20% because that's a little justifiable
DROPOUT_RATE = 0.5


# Holds my custom nueral net
class NeuralNetwork_2Layer():
    
    # This initializes all the 
    def __init__(self, inputSize, outputSize, neuronsPerLayer, learningRate = 0.1):
        
        # This is the inputSize of the data to be trained
        # Is likely going to be a number
        self.inputSize = inputSize
        
        # This is the outputSize of the data to be processed
        # It is likely to be a number
        self.outputSize = outputSize
        
        # This is the nuerons per layer, as we have to know just how many nuerons exist per layer
        # Fairly positively assuming it's going to be a matrix
        self.neuronsPerLayer = neuronsPerLayer


        # This is the learningRate, it should be assumed it is one for the entire time
        # It is likely to be a number
        self.lr = learningRate
        
        # TODO: Figure out what this does
        # This is the first layers random sample of weights
        # It is likely to be an array of numbers
        self.W1 = np.random.randn(self.inputSize, self.neuronsPerLayer)
        
        # TODO: Figure out what this does
        # This is the second layers random sample of weights
        # It is likely to be an array of numbers
        self.W2 = np.random.randn(self.neuronsPerLayer, self.outputSize)



    ## Activation function 1
    def __sigmoid(self, x):
        
        # Return the value of the sigmoid function
        return 1/( 1 + np.exp(-x) )



    ## Activation function 1's derivative
    def __sigmoidDerivative(self, x):
        
        # It's assumed that the sigmoid function is called on x before
        # it gets to this point
        # Returns the derivative of the sigmoid
        return self.__sigmoid(x)*(1-self.__sigmoid(x))



    # Activiation function 2
    def __relu(self, x):
        
        # ReLU functions is the function that operates on a value
        # Giving it value if it is greater than 0, else it is just 0
        
        # If x is greater than 0, then the value is x
        #if x > 0:
            #return x
        
        # Else the value is 0
        #else:
            #return 0
            return np.maximum( 0 , x )
    


    # The relu derivative for multiple elements
    def __reluDerivative( self , x ):

        return np.array( [ self.__reluDerivativeSingleElement(xi) for xi in x ] )
    
    

    # Activation function 2's derivative for a single element
    def __reluDerivativeSingleElement(self, x):
        
        # Since relu technically doesn't have a completey continuous
        # derivative, this gives it's derivative a value at 0

        # If the value is greater than 0, then the value is 1 
        if x > 0:

            # Return 1
            return 1

        # Else, the value is 0
        else:

            # Return 0
            return 0
    


    #  Calculates the loss function, current is mse
    #
    #> Inputs: self - the class needs a reference to itself
    #>         y ( value ) - the expected outcome value
    #>         vals ( array of values ) - the values you which to find the loss on
    #
    #> Outputs: MSE ( value ) - the Mean Squared error with respect to the expected 
    #>                          value for the data
    #
    def __loss(self, y, vals):
        
        # Duplicate the data set
        operatedValues = vals
        
        # Subtract the expected outcome value from each value
        operatedValues = operatedValues - y
        
        # Square each value
        operatedValues = np.square(operatedValues)
        
        # Divide each value by 2
        operatedValues = operatedValues/2
        
        #Return the full sum
        return sum(operatedValues)



    #  Calculates the loss function, current is mse
    #
    #> Inputs: self - the class needs a reference to itself
    #>         y ( value ) - the expected outcome value
    #>         vals ( array of values ) - the values you which to find the loss on
    #
    #> Outputs: Sum ( value ) - the sum of each data member minus the expected value
    #
    #def __lossDerivative(self, y, vals):
        
        # Duplicate the data set
        #operatedValues = vals

        # Subtract the expected outcome value from each value
        #operatedValues = operatedValues - y
        
        # Return the full sum
        #return sum(operatedValues)




    ## Batch generator for mini-batches. Not randomized.
    #> Inputs: self - the class needs to pass itself as reference
    #>         l - 
    #>         n - the size of the minibatches
    #
    #> Outputs: iterator (values) - the values for the batch
    #
    def __batchGenerator(self, l, n):
        
        # Iterate through l generating a batch of i to i + n for each item in l
        for i in range(0, len(l), n):
            
            # Return the batch of items from i to n
            yield l[i : i + n]



    ## Training with backpropagation.
    #> Inputs:  self - the class needs to pass itself as reference
    #>          xVals ( array of values ) - the input values to train on
    #>          yVals ( array of values ) - the expected output values to train on
    #>          epochs -
    #>          minibatches ( boolean ) - defines wether we should use minibatches
    #>          mbs ( value ) - the size of minibatches we want to use
    #>          
    def train(self, xVals, yVals, epochs = CUSTOM_EPOCHS, minibatches = True, mbs = MINIBATCHES):
        #print("Size of the xVals %s" % str(xVals.shape))
        #print("Size of the yVals %s" % str(yVals.shape))
        #print()
        #print()


        # Make i amount of epochs and iterate through them
        for i in range(epochs):
            
            # Make a batch for the input values
            xBatch = self.__batchGenerator( xVals , mbs )
            
            # Make a batch for the output values
            yBatch = self.__batchGenerator( yVals , mbs )

            for j in range( int( len( xVals )/mbs ) ):
                
                # Makes minibatches for the input and output values
                # Allows us to iterate over the next minibatch
                # Is a sub tensor for both
                xMiniBatch = next( xBatch )
                yMiniBatch = next( yBatch )

                # Calculates the forward pass over the input values for 2 layers
                # Is a pair of tensors of value
                layer1out, layer2out = self.__forward( xMiniBatch )
        
                # Calculates the layer 2 error
                # Is a tensor of values
                layer2error = yMiniBatch - layer2out
        
                # Calculates the layer 2 delta
                # layer2delta is a tensor of values

                # Case for if the activation function is the sigmoid
                if FUNCTION == "sigmoid":
                    
                    layer2delta = layer2error*self.__sigmoidDerivative(layer2out)
                
                # Case for if the activation function is the 
                elif FUNCTION == "relu":
                    
                    layer2delta = layer2error*self.__reluDerivative(layer2out)
                
                # Default case for the activation function
                else:
                    print("Get an activation function.")
        
        
        
                # Calculates the layer 1 error
                layer1error = np.dot( layer2delta , self.W2.T )
                

                # Calculates the layer 1 delta
                # Case for sigmoid activation function
                if FUNCTION == "sigmoid":
                    layer1delta = layer1error*self.__sigmoidDerivative( layer1out )
                
                # Case for relu activation function
                elif FUNCTION == "relu":
                    layer1delta = layer1error*self.__reluDerivative( layer1out )

                # Default case for activation function
                else:
                    print("Choose an activation function!!!!")

                
                # Calculates the Adjustement that we end up adding to each weight
                # for weight set 1
                layer1adjustment = np.dot( xMiniBatch.T , layer1delta )*self.lr

                
                # Calculates the Adjustement that we end up adding to each weight
                # for weight set 2
                layer2adjustment = np.dot( layer1out.T , layer2delta )*self.lr
                
                
                # Add the add the justment to each layer
                self.W2 = self.W2 + layer2adjustment
                self.W1 = self.W1 + layer1adjustment
                
                



    ## Forward pass.
    #> Inputs: self - this methods needs the reference of the class
    #>         input - the items that you desire to have train and get the
    #>                 values of on a first pass
    #>
    def __forward(self, input):
         
        # Do a pass over the first layer 
        
        # Case for if the activation function is sigmoid
        if FUNCTION == "sigmoid":
            layer1 = self.__sigmoid( np.dot( input , self.W1 ) )
            
        # Case for if the activation function is relu
        elif FUNCTION == "relu":
            layer1 = self.__relu( np.dot( input , self.W1 ) )
            
        # Default case for your activation function
        else:
            print("Your shit out of luck m8!")
               
        #print("Size of Layer 1 in forward%s"%str(layer1.shape))
        
        
        # Do a pass over the second layer
        
        # Case for if the activation function is sigmoid
        if FUNCTION == "sigmoid":
            layer2 = self.__sigmoid( np.dot( layer1 , self.W2 ) )
            
        # Case for if the activation function is relu
        elif FUNCTION == "relu":
            layer2 = self.__relu( np.dot( layer1 , self.W2) )
            
        # Default case for the activation function
        else:
            print("Your shit out of luck again m8!")
            
            
        #print("Shape of layer2 %s"%str(layer2.shape))
        
        
        # Returns a pair of values
        return layer1, layer2



    ## Predict.
    def predict(self, xVals):
        
        # What the fuck, I didn't even know this was legal in a coding language
        # So turns out this is legal if you want to not store the first item in the pair
        # and only care about shitty coding standards. So only layer2 gets assigned 
        # a value
        # 
        # layer2 is a tensor of values
        _, layer2 = self.__forward(xVals)
        
        # Convert the layer2 to a One-Hot array
        yValues = []

        # For every entry in the second layer 
        for entry in layer2:

            # Instantiate a One-Hot array temp
            pred = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            
            # Gets the index value that we are going to put the one hot array in
            index = np.argmax(entry)
            
            # Set the part of the One-Hot array to one
            pred[index] = 1

            # Adds the entry to the array
            yValues.append(pred)

        # Return a python array of the predicted values as One-Hot arrays
        return np.array(yValues)



## Classifier that just guesses the class label.
def guesserClassifier(xTest):
    
    # Instantiate the answers as an array
    # Will be an array of One-Hot arrays
    ans = []
    
    # Randomly create entries for the array based off of randomness
    for entry in xTest:

        # Instantiate a One-Hot array
        pred = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        
        # Set one of it's values to 1
        pred[random.randint(0, 9)] = 1
        
        # Add the entry to the answers array
        ans.append(pred)
    
    # We return a copy of the array as the array will be 
    # destroyed when we are done with it
    return np.array(ans)




#=========================<One-Hot Conversion Functions>========================




# Converts an array of values to a one hot array
def convertToOneHot(data):
    # Instantiate the answers as an array
    # Will be an array of One-Hot arrays
    ans = []

    # Randomly create entries for the array based off of randomness
    for entry in data:

        # Instantiate a One-Hot array
        pred = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        # Set one of it's values to 1
        pred[random.randint(0, 9)] = 1

        # Add the entry to the answers array
        ans.append(pred)

    # We return a copy of the array as the array will be
    # destroyed when we are done with it
    return np.array(ans)





# Converts a set of one hot arrays to value based
# arrays. So [0, 1, 0] gets converted to 1
def convertFromOneHot(data):
    # Instantiate the answers as an array
    # Will be an array of One-Hot arrays
    ans = []

    # Randomly create entries for the array based off of randomness
    for entry in data:
        value = -1

        # Convert an entry from a one hot to a number
        for i in range(entry.size):
            if entry[i] == 1:
                value = i
                break


        # Add the entry to the answers array
        ans.append(value)

    # We return a copy of the array as the array will be
    # destroyed when we are done with it
    return np.array(ans)




# Converts an array of category indices (assumed 0 to n in range)
# to a confusion matrix
def makeConfusionMatrix(yPred, yActu):
    # Instantiate a matrix to hold the answers
    matrix = np.zeros((10,10))
    matrix = matrix.astype(int)

    # Iterate through the data and give the matching 
    # sums for each category
    #
    # The x-value will be the predicted value 
    # and the y value will be the expected value
    for i in range(yPred.size):
        matrix[yPred[i]][yActu[i]] = matrix[yPred[i]][yActu[i]] + 1

    
    # Return the matrix
    # I don't know why, just did want the transpose instead
    return matrix.T




# Predicts the precision, recall and f1Score of an nxn matrix where n is the categories
def predictF1Scores(matrix):
    
    if DEBUG == 1:
        print("The shape of the matrix is %s"%str(matrix.shape))
        print("The size of the matrix is %s"%str(matrix.shape[0]))
    # Holds all the matrix values
    recall = []
    precision = []
    f1Score = []
    
    # This is the total combined matrix that we get at the end
    combinedMatrix = []

    # This is required for very simple calculation of precision
    transpose = matrix.T

    # Iterate through the matrix and calculate the recall,
    # the precision and the f1Score for each item
    # Each item will be rounded to the 3rd decimal place
    for i in range(matrix.shape[0]):
        
        if DEBUG == 1:
            print("i is %s"%str(i))

        # Calculates the recall score for category i
        recall.append(round(matrix[i][i]/sum(matrix[i]),3))

        # Calculates the precision score for category i
        precision.append(round(transpose[i][i]/sum(transpose[i]),3))

        # Calculates the f1 score for category i
        f1Score.append(round(2*recall[i]*precision[i]/(precision[i]+recall[i]),3))



    # Return the recall, the precision and the f1Score matrices as a combined numpy matrix
    combinedMatrix.append(recall)
    combinedMatrix.append(precision)
    combinedMatrix.append(f1Score)

    return np.array(combinedMatrix)


#=========================<Pipeline Functions>==================================

# Retrieves the dataset from MNIST, prints out it's size and shape then returns the dataset
# as a pair of pair of arrays
def getRawData():
    
    # Recover the mnist dataset from Keras
    mnist = tf.keras.datasets.mnist
    
    # xTrain: Large quantity of 28x28 hand drawn single digits
    #         Input training data
    # yTrain: Large quantity of actual value of digit for xTrain
    #         Output expected data
    # xTest: Large quantity of testing 28x28 hand drawn single digits
    #        Input testing data
    # yTest: Large quantity of actual value of digit for xTest
    #        Output expected data
    (xTrain, yTrain), (xTest, yTest) = mnist.load_data()
    
    # Prints the size and shape of the input training data, assumed to be (60k, 28, 28)
    print("Shape of xTrain dataset: %s." % str(xTrain.shape))
    
    # Prints the size and shape of the output training data, assumed to be (60k, )
    print("Shape of yTrain dataset: %s." % str(yTrain.shape))
    
    # Prints the size and shape of the input test data, assumed to be (10k, 28, 28)
    print("Shape of xTest dataset: %s." % str(xTest.shape))
    
    # Prints the size and shape of the output test data, assumed to be (10k, )
    print("Shape of yTest dataset: %s." % str(yTest.shape))
    
    # Returns our raw data from the mnist set
    return ((xTrain, yTrain), (xTest, yTest))


# Converts the input data from byte size to floating point value between [0.0,1.0],
# then converts the input data from an array of images to an array of arrays,
# then converts the output data from integers to One-Hot arrays
# prints off the transformed dataset shape and size 
# then returns a <<array,array<array>>,<<array>,<array<array>>>
def preprocessData(raw):
    # Calculates the Adjustement that we end up adding to each weight
                # fo weight 1

    # Make variable place holders for the pair of pairs in raw
    ((xTrain, yTrain), (xTest, yTest)) = raw            


    ##TODONE: Add range reduction here (0-255 ==> 0.0-1.0).
    
    # Divides every input train value by 255
    xTrain = xTrain/MAX_COLOUR_VALUE
   


    # Convert the training images to arrays
    # Used to be ( 60k , 28 , 28 ) to ( 60k , 784 )
    xTrain = xTrain.reshape( TRAINING_SIZE , IMAGE_SIZE )

    
    # Divides every input test value by 255
    xTest = xTest/MAX_COLOUR_VALUE

    # Convert the test images to arrays
    # Used to be ( 10k , 28 , 28 ) to ( 10k , 784 )
    xTest = xTest.reshape( TEST_SIZE , IMAGE_SIZE )



    # yTrainP is a transform on yTrain to turn numbers into One-Hot arrays
    yTrainP = to_categorical(yTrain, NUM_CLASSES)

    # yTestP is a transorm on yTest to turn numbers into One-Hot arrays
    yTestP = to_categorical(yTest, NUM_CLASSES)
        
    # Prints the size and shape of the input training data, assumed to be (60k, 28, 28)
    print("New shape of xTrain dataset: %s." % str(xTrain.shape))
    
    # Prints the size and shape of the output training data, assumed to be (60k, )
    print("New shape of xTest dataset: %s." % str(xTest.shape))
    
    # Prints the size and shape of the transformed input testing data, assumed to be (10k, 28, 28)
    print("New shape of yTrain dataset: %s." % str(yTrainP.shape))
    
    # Prints the size and shape of the transformed output testing data, assumed to be (10k, )
    print("New shape of yTest dataset: %s." % str(yTestP.shape))
    
    # Returns the old training data with the outputs
    return ((xTrain, yTrainP), (xTest, yTestP))


#TODO: Figure out what the fuck this does and write the code for it
# Input: a pair of training data that has the training input and 
#       the expected output value for each input indexed respectively
# 

def trainModel(data):
    
    # xTrain is an array of flattened arrays representing each input image
    # yTrain is an array of One-Hot arrays representing each expected output
    xTrain, yTrain = data
    
    print("Shape of yTrain %s"%str(yTrain.shape))

    # Case for a randomized guessing algorithm
    if ALGORITHM == "guesser":
        return None   
        #Guesser has no model, as it is just guessing.
    
    # Case for my custom nueral net class
    elif ALGORITHM == "custom_net":
        
        # Debugger that tells the user what's going on
        print("Building and training Custom_NN.")
        
        # Prints the status of the function
        #print("Not yet implemented.")                   
        print("Implemented.")


        #TODO: Write code to build and train your custom neural net.
        
        myModel = NeuralNetwork_2Layer( IMAGE_SIZE , ONE_HOT_SIZE , 512)
        
        myModel.train( xTrain , yTrain )
        
        
        # TODO: Figure out what we need to return
        return myModel
    
    # Case for my Keras based nueral net
    elif ALGORITHM == "tf_net":
        
        # Debugger that tells the user what's going on
        print("Building and training TF_NN.")
        
        # Prints the status of the function
        #print("Not yet implemented.")
        print("Implemented.")


        #TODO: Write code to build and train your keras neural net.
        # Instantiates our model
        tfModel = tf.keras.Sequential()
        
        # Instantiates the lossType for the model
        lossType = tf.keras.losses.CategoricalCrossentropy()
        
        # This is the optimizer with which we will compile the nueral net
        opt = tf.train.AdamOptimizer()
        
        # The input data size
        inShape = (IMAGE_SIZE,)
        
        # This adds the first layer
        #tfModel.add( keras.layers.Dense( NUERON_COUNT_PER_LAYER , kernel_initializer = 'he_uniform',  input_shape = inShape , activation = tf.nn.relu ) )
        tfModel.add( keras.layers.Dense( NUERON_COUNT_PER_LAYER ,  input_shape = inShape , activation = tf.nn.relu ) )

        # Sprinkle a little dropout in there
        tfModel.add( keras.layers.Dropout( DROPOUT_RATE/2 ) )

        # This adds the second layer
        #tfModel.add( keras.layers.Dense( NUERON_COUNT_PER_LAYER, activation = tf.nn.relu ) )

        # Add some more dropout
        #tfModel.add( keras.layers.Dropout( DROPOUT_RATE/2 ) )
        
        # This adds a third layer
        #tfModel.add( keras.layers.Dense( NUERON_COUNT_PER_LAYER/4, activation = tf.nn.sigmoid ) )

        # Hit this nueral net with that C- in 252 feeling
        #tfModel.add(keras.layers.Dropout( DROPOUT_RATE/2 ) ) 

        # This adds the final layer
        tfModel.add( keras.layers.Dense( ONE_HOT_SIZE , activation = tf.nn.softmax ) )

        # Compile the nueral net
        tfModel.compile( optimizer = opt , loss = lossType , metrics = ['accuracy'] )
        
        # Trains the data based on the training data provided
        tfModel.fit( xTrain , yTrain, epochs = TF_EPOCHS , batch_size = 32 )
        
        print("Gets here")
        
        

        #TODO: Figure out what we need to return
        return tfModel
    
    # You really done goofed somehow if you got here
    else:
        
        # Spits out an error if the algorithm type is a bad type
        raise ValueError("Algorithm not recognized.")


# Runs the model on the code to test the outputs of the fully
# trained nueral net
def runModel(data, model):
   


    # Case where we just want to guess the right answer
    if ALGORITHM == "guesser":
        
        # Return the call to the guesser classifier method
        # that just randomly guesses the answer
        return guesserClassifier(data)
    
    # Case where we want to test the custom nueral net 
    elif ALGORITHM == "custom_net":
        
        # Debugger that tells the user what's going on
        print("Testing Custom_NN.")
        
        # Prints the status of the function
        #print("Not yet implemented.")
        print("Implemented.")
        
        # Get the predictions from the model

        dataset = model.predict(data)
        
        
        # Return the dataset
        return dataset
    
    # Case where we want to test the Keras model
    elif ALGORITHM == "tf_net":
        
        # Debugger that tells the user what's going on
        print("Testing TF_NN.")
        
        # Prints the status of the function
        #print("Not yet implemented.")
        print("Implemented.")
        

        #TODO: Write code to run your keras neural net.
        dataset = model.predict(data)
        
        if DEBUG == 1:
            print(dataset) 

        # Convert the layer2 to a One-Hot array
        yValues = []

        # For every entry in the second layer 
        for entry in dataset:

            # Instantiate a One-Hot array temp
            pred = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

            # Gets the index value that we are going to put the one hot array in
            index = np.argmax(entry)

            # Set the part of the One-Hot array to one
            pred[index] = 1

            # Adds the entry to the array
            yValues.append(pred)

        # Return a python array of the predicted values as One-Hot arrays
        return np.array(yValues)


        #print("Shape of dataset %s"%tuple(dataset))
        # Figure out what we need to return
        return dataset

    
    #You done goofed if it got here
    else:
        
        # Spits out an error if the algorithm type is a bad type
        raise ValueError("Algorithm not recognized.")




# This evaluates the results of the nueral nets, giving an F1 score
# and then outputs how good it was
# >Inputs:
#           data - the dataset from MNIST
#           preds - the dataset the model estimated the answer to be
def evalResults(data, preds):   
    
    #TODO: Add F1 score confusion matrix here.
    
    # The testing data from MNIST
    # A pair of arrays that hold the test inputs and outputs
    xTest, yTest = data
    
    # Revert them from One Hot arrays to normal arrays

    yRevert = convertFromOneHot(yTest)
    pRevert = convertFromOneHot(preds)
    
    # Print off a few new lines for spacing
    print("\n")

    # Compute the f1_score for the classes
    if DEBUG == 1:
        print("The shape of predictions: %s"%str(preds.shape))
        print(preds)

    
    
    
    # Converts the one hot arrays back to valued items
    yPred = pd.Series(pRevert, name='Predicted')
    yAct = pd.Series(yRevert, name='Actual')

    
    # This is the confusion matrix of values that we want to print off
    confusionMatrix = makeConfusionMatrix(yPred, yAct)
    
    # This is the recall, precision and f1 scores of the
    # confusion matrix that we calculated
    scores = predictF1Scores(confusionMatrix)
   
    # Prints off the axis labels
    print("X-Axis is the Actual Value and Y-Axis is the Predicted Value")

    # Prints the confusion matrix as a dataframe
    print(pd.DataFrame(confusionMatrix))
    
    # Add some spaces to free up things
    print("\n")

    # Prints off the scores as a dataframe
    print(pd.DataFrame(scores,index=['Recall','Precision','F1 Score']))


    # Print off a few new lines for spacing
    print("\n")





    # The accuracy counter of the data, assumed to be 0 without any data
    acc = 0
    
    # Go through each item in the predictions array and 
    for i in range(preds.shape[0]):
        
        # Increment acc for every right piece of data there is
        if np.array_equal(preds[i], yTest[i]):   acc = acc + 1
    
    # The accuracy is the accuracy counter divided by the count of values estimated correctly
    accuracy = acc / preds.shape[0]
    
    # Print out the algorithm type
    print("Classifier algorithm: %s" % ALGORITHM)
    
    # Print out the accuracy of the model
    print("Classifier accuracy: %f%%" % (accuracy * 100))
    
    # Print a new line
    print()



#=========================<Main>================================================


#
def main():
    
    # Set raw equal to the mnist data
    # raw: pair of pair of arrays, ie. ( ( xTrain , yTrain ) , ( xTest , yTest ) )
    raw = getRawData()
    
    # Converts the inputs to values between 0 and 1,
    # then flattens each input image and 
    # converts the output to binary matrices
    # data: pair of <array, binary matrix> pairs
    data = preprocessData(raw)
    
    # Trains the model on the training data
    # TODO: Figure out what type model is
    # model:
    model = trainModel(data[0])
    
    # Runs the model against the testing data
    # TODO: Figure out what type preds is
    # preds: 
    preds = runModel(data[1][0], model)
    
    # Evaluates the results of the predictions,
    # calculating the scores of efficiency 
    # and outputs the values to the standard out
    evalResults(data[1], preds)




if __name__ == '__main__':
    
    # Calls the main function
    main()
