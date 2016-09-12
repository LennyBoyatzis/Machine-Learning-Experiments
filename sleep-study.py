import numpy as np

# Data (hours study vs hours sleep night before test)
# Supervised regression problem

X = np.array(([3,5], [5,1], [10,2]), dtype=float)
y = np.array(([75], [82], [93]), dtype=float)

# Normalise our data by dividing by max score (100)
# Scales our data down to a value between 0 & 1

X = X/np.amax(X, axis=0)
y = y/100 # Max test score is 100

# Our network needs to have two inputs and 1 output
# Because these are the dimensions of out data
# We will use 1 hidden layer with 3 hidden units

# Synapses have a really simple job
# Take a value from their input and multiply it by a specific weight and output the result

# Neurons
# Have the job of adding together the outputs from all of their synapses and apply an activation function (sigmoid function) - these enable neural nets to model complex non-linear patterns

# Forward Propagation
class Neural_Network(object):
    def __init__(self):
        self.inputLayerSize = 2
        self.outputLayerSize = 1
        self.hiddenLayerSize = 3

        self.W1 = np.random.randn(self.inputLayerSize, self.hiddenLayerSize)
        self.W2 = np.random.randn(self.hiddenLayerSize, self.outputLayerSize)

    def forward(self, X):
        self.z2 = np.dot(X, self.W1)
        self.a2 = self.sigmoid(self.z2)
        self.z3 = np.dot(self.a2, self.W2)
        yHat = self.sigmoid(self.z3)
        return yHat

    def sigmoid(self, z):
        return 1/(1+np.exp(-z))



NN = Neural_Network()
yHat = NN.forward(X)

print "yHat -> %s" %yHat
