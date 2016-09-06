import numpy as np

# Function to map any value to a val between 0-1
# This is a sigmoid
# Run on every neuron of our network
# Useful for creating probabilities out of numbers
def nonlin(x, deriv=False):
    if(deriv==True):
        return x*(1-x)
    return 1/(1+np.exp(-x))

# Initialise input data set as a matrix
# Each row is a different training example
# Each column represents a different neuron

# input data (4 training examples with 3 input neurons each)
X = np.array([
    [0,0,1],
    [0,1,1],
    [1,0,1],
    [1,1,1],
])

# output data (4 examples with 1 output neuron each)
y = np.array([
    [0],
    [1],
    [1],
    [0]
])

# Generating random numbers in a sec
np.random.seed(1)

# Create out synapse matrices
# These are the connections between each neuron
# We have 3 layers in our network so we need to synapse matrices

syn0 = 2*np.random.random((3,4)) - 1
syn1 = 2*np.random.random((4,1)) - 1

# Training step
for j in xrange(60000):
    l0 = X
    # Matrix multiplication between each layer and its synapse
    l1 = nonlin(np.dot(l0, syn0))
    l2 = nonlin(np.dot(l1, syn1))
    l2_error = y - l2

    if (j % 10000) == 0:
        print "Error:" + str(np.mean(np.abs(l2_error)))

    # Backpropagation
    l2_delta = l2_error*nonlin(l2, deriv=True)
    l1_error = l2_delta.dot(syn1.T)
    l1_delta = l1_error * nonlin(l1, deriv=True)

    # update weights (gradient descent)
    syn1 += l1.T.dot(l2_delta)
    syn0 += l0.T.dot(l1_delta)

print "Output after training"
print l2
