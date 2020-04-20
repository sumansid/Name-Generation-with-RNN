import numpy as np
#!/usr/bin/env python
# coding: utf-8

# # Recurrent Neural Network Implementation


# Activation functions



def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)




def sigmoid(x):
    return 1 / (1 + np.exp(-x))


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


def rnn_forward(x, a0, parameters):
    """
    Arguments : 
    a0 : Initial set of activation (Zero vector)
    x : Input for every time setp (n_x,m,Tx) 
    parameters : Dicitonary that contains the weights and biases
    """
    # Cache values for backprop
    caches = []
    # Necessary dimensions for initialization
    n_x, m, T_x = x.shape
    n_y, n_a = parameters["Wya"].shape
    a = np.zeros([n_a,m,T_x])
    a_next = a0
    y_pred = np.zeros([n_y,m,T_x])
    for t in range(T_x):
        # Get 2d slice at timestep t 
        xt = x[:,:,t]
        a_next,yt_pred, cache = rnn_cell_forward(a_next, xt, parameters)
        a[:,:,t] = a_next
        y_pred[:,:,t] = yt_pred
        caches.append(cache)
    # store values needed for backward propagation in cache
    caches = (caches, x)
    
    return a, y_pred, caches

def clip_gradient(gradients, max_value):
	'''
	Clips the gradients into a given range to avpid exploding gradients
	'''
	dWaa, dWax, dWya, db, dby = gradients['dWaa'], gradients['dWax'], gradients['dWya'], gradients['db'], gradients['dby']
	for gradient in [dWax, dWaa, dWya, db, dby]:
        np.clip(gradient, a_max=maxValue, a_min=-maxValue, out=gradient)

    gradients = {"dWaa": dWaa, "dWax": dWax, "dWya": dWya, "db": db, "dby": dby}
    
    return gradients



    
    


# In[ ]:





# In[ ]:




