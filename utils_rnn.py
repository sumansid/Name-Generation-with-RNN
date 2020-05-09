import numpy as np
"""
Date : 08/05/2020
Author : Suman Sigdel
File : Utils.py contains the utilities functions for the name generation model
"""

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

## Function that initializes the parameters

def initialize_parameters(n_a, n_x, n_y):
    # Weights and Bias of hidden and output
  
    Wax = np.random.randn(n_a, n_x)*0.01 # Weight scaling
    Waa = np.random.randn(n_a, n_a)*0.01 
    Wya = np.random.randn(n_y, n_a)*0.01 
    b = np.zeros((n_a, 1))
    by = np.zeros((n_y, 1)) 
    
    parameters = {"Wax": Wax, "Waa": Waa, "Wya": Wya, "b": b,"by": by}
    
    return parameters


# ## RNN Cell forward prop


def rnn_cell_forward(a_prev, xt, parameters, vocab_size = 27):
    """
    Function that computes the forward prop for each RNN cell
    
    parameters : Dictionary containing the weights and biases
    """
    Wax = parameters["Wax"]
    Waa = parameters["Waa"]
    Wya = parameters["Wya"]
    ba = parameters["ba"]
    by = parameters["by"]
    
    a_next = np.tanh(np.dot(Waa,a_prev) + np.dot(Wax,xt)  + ba)
    yt_pred = softmax(np.dot(Wya,a_next) + by)
    
    # Cache for backpropagation
    cache = (a_next, a_prev, xt, parameters)
    
    return a_next,yt_pred, cache


# ## RNN Forward propagation

# In[14]:

def rnn_step_forward(parameters, a_prev, x):
    
    Waa, Wax, Wya, by, b = parameters['Waa'], parameters['Wax'], parameters['Wya'], parameters['by'], parameters['b']
    a_next = np.tanh(np.dot(Wax, x) + np.dot(Waa, a_prev) + b) # hidden state
    p_t = softmax(np.dot(Wya, a_next) + by) # unnormalized log probabilities for next chars # probabilities for next chars 
    
    return a_next, p_t

def rnn_step_backward(dy, gradients, parameters, x, a, a_prev):
    
    gradients['dWya'] += np.dot(dy, a.T)
    gradients['dby'] += dy
    da = np.dot(parameters['Wya'].T, dy) + gradients['da_next'] # backprop into h
    daraw = (1 - a * a) * da # backprop through tanh nonlinearity
    gradients['db'] += daraw
    gradients['dWax'] += np.dot(daraw, x.T)
    gradients['dWaa'] += np.dot(daraw, a_prev.T)
    gradients['da_next'] = np.dot(parameters['Waa'].T, daraw)
    return gradients


def rnn_forward_with_loss(X, Y, a0, parameters, vocab_size = 27):
    
    x, a, y_hat = {}, {}, {}
    a[-1] = np.copy(a0)
 
    loss = 0
    
    for t in range(len(X)):
       
        x[t] = np.zeros((vocab_size,1)) 
        if (X[t] != None):
            x[t][X[t]] = 1
        
        # Run one step forward of the RNN
        a[t], y_hat[t] = rnn_step_forward(parameters, a[t-1], x[t])
        
        # Update the loss by substracting the cross-entropy term of this time-step from it.
        loss -= np.log(y_hat[t][Y[t],0])
        
    cache = (y_hat, a, x)
        
    return loss, cache


def clip_gradient(gradients, maxValue):
    '''
    Clips the gradients into a given range to avpid exploding gradients
    '''
    dWaa, dWax, dWya, db, dby = gradients['dWaa'], gradients['dWax'], gradients['dWya'], gradients['db'], gradients['dby']
    for gradient in [dWax, dWaa, dWya, db, dby]:
        np.clip(gradient, a_max=maxValue, a_min=-maxValue, out=gradient)

   
    gradients = {"dWaa": dWaa, "dWax": dWax, "dWya": dWya, "db": db, "dby": dby}
    
    return gradients


def rnn_backward(X, Y, parameters, cache):
    
    gradients = {}
   
    (y_hat, a, x) = cache
    Waa, Wax, Wya, by, b = parameters['Waa'], parameters['Wax'], parameters['Wya'], parameters['by'], parameters['b']
    
    gradients['dWax'], gradients['dWaa'], gradients['dWya'] = np.zeros_like(Wax), np.zeros_like(Waa), np.zeros_like(Wya)
    gradients['db'], gradients['dby'] = np.zeros_like(b), np.zeros_like(by)
    gradients['da_next'] = np.zeros_like(a[0])
   
    for t in reversed(range(len(X))):
        dy = np.copy(y_hat[t])
        dy[Y[t]] -= 1
        gradients = rnn_step_backward(dy, gradients, parameters, x[t], a[t], a[t-1])
   
    
    return gradients, a
    
    
def update_parameters(parameters, gradients, learning_rate):
    lr = learning_rate
    parameters['Wax'] += -lr * gradients['dWax']
    parameters['Waa'] += -lr * gradients['dWaa']
    parameters['Wya'] += -lr * gradients['dWya']
    parameters['b']  += -lr * gradients['db']
    parameters['by']  += -lr * gradients['dby']
    return parameters
    
def optimize(X, Y, a_prev, parameters, learning_rate = 0.01): 
    
    loss, cache = rnn_forward_with_loss(X,Y, a_prev, parameters)
    gradients, a = rnn_backward(X, Y, parameters, cache)
    gradients = clip_gradient(gradients,maxValue=5)
    parameters = update_parameters(parameters,gradients,learning_rate)

    return loss, gradients, a[len(X)-1]

def print_sample(sample_idx, idx_to_char):
    txt = ''.join(idx_to_char[idx] for idx in sample_idx)
    txt = txt[0].upper() + txt[1:]  # capitalize first character 
    print ('%s' % (txt, ), end='')

def smooth(loss, cur_loss):
    return loss * 0.999 + cur_loss * 0.001

# Defining a model
def model(data, filename, idx_to_char, char_to_idx, num_iter = 35000, n_a = 50, nepali_name = 10, vocab_size=27):
    
    n_x, n_y = vocab_size, vocab_size
    loss = -np.log(1.0/vocab_size)*nepali_name
    loss_points = []
    
    parameters = initialize_parameters(n_a, n_x, n_y)
    with open(filename) as f:
        training_examples = f.readlines()
    training_examples = [x.lower().strip() for x in training_examples]
    # Initialize the hidden state 
    a_prev = np.zeros((n_a, 1))
    np.random.shuffle(training_examples)
    
    for j in range(num_iter):
        # Mod ie. % returns back to index 0 once we reach end of the examples
        idx = j % len(training_examples)
        # Get an example from idx 
        single_training_example = training_examples[idx]
        # Get all the characters from the single training example
        single_training_character = [char for char in single_training_example]
        # Get the indexes of the characters in the single training example
        single_example_idx = [char_to_idx[ch] for ch in single_training_character]
        # None prepended to set the input vector to 0 vector
        X = [None] + single_example_idx
        idx_newline = char_to_idx["\n"]
        Y = X[1:] + [idx_newline]
        
        curr_loss, gradients, a_prev = optimize(X,Y,a_prev,parameters,learning_rate=0.01)
        loss = smooth(loss, curr_loss)
        
        if j % 1000 == 0:
            
            print('Iteration: %d, Loss: %f' % (j, loss) + '\n')
            loss_points.append(loss)
            # Sampling the names
            for name in range(nepali_name):

                # Sample indices and print them
                sampled_indices = sample(parameters, char_to_idx)
                print_sample(sampled_indices, idx_to_char)
      
            print('\n')
        
    return loss_points
