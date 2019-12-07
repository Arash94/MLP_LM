

To Run the code, please run $python ann.py in terminal.

This network approximates a non-linear function of the form f: R3 -> R using a feed forward
Artificial Neural Netwrok (ANN). In leu of taking of advantage of Gradient Descent, we exercise 
a new Optimizer to update the weights and minimize the loss function. LM is much faster at finding 
a minima than gradient descent.

--The network is setup by specifying the network layout.
    e.g. [3,3,1] means 3 inputs, 3 hidden units and 1 output. Hence we're mapping from R3 to R.

-->>Try adding more hidden units of size 3 neurons per layer and observe better generalization of the network
        note*: Each layer has a bias

--The activation is also specified as tanh to prevent the network from learning linear relationships.

--The jacobian matrix for the LM optimizer is computed on every backward pass on a per layer basis.
--




some test data to check manually by hand:
    ### Train the network ###
    # a =np.array([ 9.15831663, -0.14028056,  2.2244489 ])
    # b =np.array([3.74749499,  3.10621242,  1.86372745])
    # a = a.reshape(-1,1).T
    # b = b.reshape(-1,1).T
    # x = np.vstack((a,b))
    # y = np.array([[9.34193035,   3.01480717]]).T



