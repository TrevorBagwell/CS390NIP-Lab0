
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
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
ALGORITHM = "guesser"
#ALGORITHM = "custom_net"
#ALGORITHM = "tf_net"




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
        return x*(1-x)



    # Activiation function 2
    def __relu(self, x):
        
        # ReLU functions is the function that operates on a value
        # Giving it value if it is greater than 0, else it is just 0
        return max( 0 , x )



    # Activation function 2's derivative
    def __reluDerivative(self, x):
        
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
    def __loss(self, y, vals):
        
        # Duplicate the data set
        operatedValues = vals

        # Subtract the expected outcome value from each value
        operatedValues = operatedValues - y
        
        # Return the full sum
        return sum(operatedValues)




    ## Batch generator for mini-batches. Not randomized.
    #> Inputs: self - the class needs to pass itself as reference
    #>         l - 
    #>         n - the size by which we want to split the minibatches
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
    def train(self, xVals, yVals, epochs = 100000, minibatches = True, mbs = 100):
        
        # Calculates the forward pass over the input values for 2 layers
        # Is a pair of values
        layer1out, layer2out = self.__forward( xVals )
        
        # Calculates the layer 2 error
        # Is a value
        layer2error = self.__lossDerivative(y,layer2out)


        # Calculates the layer 2 delta
        # layer2delta is a value
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
        # Figure out what w^2 is
        #layer1error = 

        #pass                                   
        
        ##TODO: Implement backprop. allow minibatches. mbs should specify the size of each minibatch.
        


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




        # Do a pass over the second layer

        # Case for if the activation function is sigmoid
        if FUNCTION == "sigmoid":
            layer2 = self.__sigmoid( np.dot( layer1 , self.W2 ) )
        
        # Case for if the activation function is relu
        elif FUNCTION == "relu":
            layer2 = self.__relu( np.dot( layer2 , self.W2) )

        # Default case for the activation function
        else:
            print("Your shit out of luck again m8!")

        # Returns a pair of values
        return layer1, layer2



    ## Predict.
    def predict(self, xVals):
        
        # What the fuck, I didn't even know this was legal in a coding language
        _, layer2 = self.__forward(xVals)
        
        #TODO: Figure out why we return layer2
        return layer2



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
    
    # Case for a randomized guessing algorithm
    if ALGORITHM == "guesser":
        return None   
        #Guesser has no model, as it is just guessing.
    
    # Case for my custom nueral net class
    elif ALGORITHM == "custom_net":
        
        # Debugger that tells the user what's going on
        print("Building and training Custom_NN.")
        
        # Prints the status of the function
        print("Not yet implemented.")                   
        #print("Implemented.")


        #TODO: Write code to build and train your custom neural net.
        
        
        
        # TODO: Figure out what we need to return
        return None
    
    # Case for my Keras based nueral net
    elif ALGORITHM == "tf_net":
        
        # Debugger that tells the user what's going on
        print("Building and training TF_NN.")
        
        # Prints the status of the function
        print("Not yet implemented.")
        #print("Implemented.")


        #TODO: Write code to build and train your keras neural net.
        
        
        #TODO: Figure out what we need to return
        return None
    
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
        print("Not yet implemented.")
        #print("Implemented.")


        #TODO: Write code to run your custom neural net.
        
        # Figure out what we need to return
        return None
    
    # Case where we want to test the Keras model
    elif ALGORITHM == "tf_net":
        
        # Debugger that tells the user what's going on
        print("Testing TF_NN.")
        
        # Prints the status of the function
        print("Not yet implemented.")
        #print("Implemented.")
        

        #TODO: Write code to run your keras neural net.
        

        # Figure out what we need to return
        return None
    
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
