from sklearn import *
from numpy import *
from matplotlib import pyplot as plt



# Generate a dataset and plot it
random.seed(0)
X, y = datasets.make_moons(200, noise=0.20)
plt.scatter(X[:,0], X[:,1], s=40, c=y, cmap=plt.get_cmap('Spectral'))


num_examples = len(X) # training set size
nn_input_dim = 2 # input layer dimensionality
nn_output_dim = 2 # output layer dimensionality

# Gradient descent parameters (picked by hand)
epsilon = 0.01 # learning rate for gradient descent
reg_lambda = 0.01 # regularization strength

# Helper function to evaluate the total loss on the dataset
def calculate_loss(model):
    W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
    # Forward propagation to calculate our predictions
    z1 = X.dot(W1) + b1
    a1 = tanh(z1)
    z2 = a1.dot(W2) + b2
    exp_scores = exp(z2)
    probs = exp_scores / sum(exp_scores, axis=1, keepdims=True)
    # Calculating the loss
    correct_logprobs = -log(probs[range(num_examples), y])
    data_loss = sum(correct_logprobs)
    # Add regularization term to loss (optional)
    data_loss += reg_lambda/2 * (sum(square(W1)) + sum(square(W2)))
    return 1./num_examples * data_loss

# Helper function to predict an output (0 or 1)
def predict(model, x):
    W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
    # Forward propagation to calculate our predictions
    z1 = X.dot(W1) + b1
    a1 = tanh(z1)
    z2 = a1.dot(W2) + b2
    exp_scores = exp(z2)
    probs = exp_scores / sum(exp_scores, axis=1, keepdims=True)
    return argmax(probs, axis=1)

# This function learns parameters for the neural network and returns the model.
# - nn_hdim: Number of nodes in the hidden layer
# - num_passes: Number of passes through the training data for gradient descent
# - print_loss: If True, print the loss every 1000 iterations
def build_model(nn_hdim, num_passes=20000, print_loss=False):

    # Initialize the parameters to random values. We need to learn these
    random.seed(0)
    W1 = random.randn(nn_input_dim, nn_hdim) / sqrt(nn_input_dim)
    b1 = zeros((1, nn_hdim))
    W2 = random.randn(nn_hdim, nn_output_dim) / sqrt(nn_hdim)
    b2 = zeros((1, nn_output_dim))

    # This is what we return at the end
    model = {}

    # Gradient descent. For each batch...
    for i in range(0, num_passes):

        # Forward propagation
        z1 = X.dot(W1) + b1
        a1 = tanh(z1)
        z2 = a1.dot(W2) + b2
        exp_scores = exp(z2)
        probs = exp_scores / sum(exp_scores, axis=1, keepdims=True)

        # Backpropagation
        delta3 = probs
        delta3[range(num_examples), y] -= 1
        dW2 = (a1.T).dot(delta3)
        db2 = sum(delta3, axis=0, keepdims=True)
        delta2 = delta3.dot(W2.T) * (1 - power(a1, 2))
        dW1 = dot(X.T, delta2)
        db1 = sum(delta2, axis=0)

        # Add regularization terms (b1 and b2 don't have regularization terms)
        dW2 += reg_lambda * W2
        dW1 += reg_lambda * W1

        # Gradient descent parameter update
        W1 += -epsilon * dW1
        b1 += -epsilon * db1
        W2 += -epsilon * dW2
        b2 += -epsilon * db2

        # Assign new parameters to the model
        model = {'W1':W1, 'b1':b1, 'W2':W2, 'b2':b2}

        # Optionally print the loss.
        # This is expensive because it uses the whole dataset, so we don't want to do it too often
        if print_loss and i % 1000 == 0:
            print("Loss after iteration %i: %f" %(i, calculate_loss(model)))

    return model

# Build a model with a 3-dimensional hidden layer
model = build_model(3, print_loss=True)

# Plot the decision boundary
plot_decision_boundary = lambda x:predict(model, x)

plt.title("Decision Boundary for hidden layer size 3")
plot_decision_boundary
plt.show()


"""
plt.figure(figsize=(16, 32))
hidden_layer_dimensions = [1, 2, 3, 4, 5, 20, 50]
for i, nn_hdim in enumerate(hidden_layer_dimensions):
    plt.subplot(5, 2, i+1)
    plt.title('Hidden Layer size %d' % nn_hdim)
    model = build_model(nn_hdim)
    plot_decision_boundary
plt.show()
"""
