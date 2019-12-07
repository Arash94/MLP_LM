import numpy as np 

config = {}
config['layer_specs'] = [3, 3, 1]
config['activation'] = 'tanh' # Takes values 'sigmoid', 'tanh' or 'ReLU'; denotes activation function for hidden layers
config['epochs'] = 200 # Number of epochs to train the model
config['early_stop'] = True  # Implement early stopping or not
config['early_stop_epoch'] = 5  # Number of epochs for which validation loss increases to be counted as overfitting
config['batch_size'] = 50
#___PLEASE CHOOSE THE OPTIMIZER__#
# LM => stands fpr 'Levenberg_Marquardt' algorithm
# GD =>stands for 'Gradient_Descent' algorithm
config['optimizer'] = 'LM'
config['gamma'] = 0.01
config['L2 const'] = 0.001
config['learning_rate'] = 0.001 # Learning rate of gradient descent algorithm


def generate_input(BOUND, NUM_SAMPLE):
    
    x1 = np.linspace(BOUND,-BOUND,NUM_SAMPLE, dtype=float)
    x2 = np.linspace(BOUND,-BOUND,NUM_SAMPLE, dtype=float)
    x3 = np.linspace(BOUND,-BOUND,NUM_SAMPLE, dtype=float)
    x1 = x1[np.random.permutation(x1.shape[0])].reshape(1,NUM_SAMPLE)
    x2 = x2[np.random.permutation(x2.shape[0])].reshape(1,NUM_SAMPLE)
    x3 = x3[np.random.permutation(x3.shape[0])].reshape(1,NUM_SAMPLE)
    
    target = np.multiply(x1,x2)+x3
    train = np.vstack((np.vstack((x1,x2)),x3))
    print("size of each input:{} ".format(train.T[0].shape))

    return target.T , train.T

class Activation:
    def __init__(self, activation_type = "sigmoid"):
        self.activation_type = activation_type
        self.x = None # Save the input 'x' for sigmoid or tanh or ReLU to this variable since it will be used later for computing gradients.

    def forward_pass(self, a):
        
        return self.tanh(a)

    def backward_pass(self, delta):
        grad = self.grad_tanh()  
        return grad

    def tanh(self, x):
        """
        code for tanh activation function that takes in a numpy array and returns a numpy array.
        """
        self.x = x
        return np.tanh(x)

    def grad_tanh(self):
        """
        code for gradient through tanh activation function that takes in a numpy array and returns a numpy array.
        """
        return (1 - np.power(np.tanh(self.x), 2))


class Layer():
    def __init__(self, config, layer_index):
        np.random.seed(42)

        in_units = config['layer_specs'][layer_index]
        out_units = config['layer_specs'][layer_index + 1]
        self.selelct_optizmizer = config['optimizer']
        self.w = 1 / (in_units + out_units) * np.random.randn(in_units, out_units)

        self.b = np.zeros((1, out_units)).astype(np.float32)    # Bias
        self.temp_w = self.w
        self.temp_b = self.b
        self.gamma = config['gamma']
        self.learning_rate = config.get('learning_rate', 0)
        self.x = None    # Save the input to forward_pass in this
        self.a = None    # Save the output of forward pass in this (without activation)
        self.d_x = None    # Save the gradient w.r.t x in this
        self.d_w = None    # Save the gradient w.r.t w in this
        self.d_b = None    # Save the gradient w.r.t b in this

    def forward_pass(self, x):
        """
        Write the code for forward pass through a layer
        """

        self.x = x
        self.a = x @ self.w + self.b
        # print('shape of x'+ str(x.shape))
        # print('shape of weight'+ str(self.w.shape))
        # print(self.w)
        return self.a

    def backward_pass(self, delta):
        """
        backward pass. This takes in gradient from its next layer as input,
        computes gradient for its weights and the delta to pass to its previous layers.
        The Jacbian is stored here
        """
        self.d_x = self.w.T #forward gradient!
        self.d_w = self.x.T @ delta 
        self.d_b = np.sum(delta, axis=0, keepdims=True)

        new_delta = delta @ self.d_x
        # print("b")
        # print(self.d_b.shape)
        # print("delta")
        # print(delta.shape)
        I1 = np.eye(self.w.shape[0]) #creating the dynamic shape I matrix
        I2 = np.eye(self.b.shape[0])

        """
        after delta has been passed back,
        implementing Levenberg Marquardt optimizer to compute the new weights going out of this layer (lamda varies, adaptive learning)
        wt+1 for this hidden layer:
        """
        if self.selelct_optizmizer == 'LM':
            self.temp_w = self.w - np.linalg.inv(self.d_w.T @ self.d_w + self.gamma*I1) @ self.d_w @ self.d_x
            #self.temp_b = self.b - np.linalg.inv(self.d_b.T @ self.d_b + self.gamma*I2) @ self.d_b @ self.d_x

        if self.selelct_optizmizer == 'GD':
            self.w += self.learning_rate * self.d_w
            self.b += self.learning_rate * self.d_b


        return new_delta

    def weight_loss_func(self):
        """
        L2 loss function for weights
        """
        return np.linalg.norm(self.w,ord=2)**2
    
    #adjusting the gamma parameter for LM to adapt to the learning enviroment
    def step(self, accept=True):
        if accept:
            self.w = self.temp_w  
            self.b = self.temp_b
            self.gamma = 0.8*self.gamma
        else:
            self.gamma = 1.2*self.gamma


    """
        These two functions will save the best best weights of the network when early stop is called
    """
    def save_weights(self):
        self.best_w = self.w
        self.best_b = self.b

    def use_best_weights(self):
        self.w = self.best_w
        self.b = self.best_b

class Neuralnetwork():
    def __init__(self, config):
        self.layers = []
        self.x = None        # Save the input to forward_pass in this
        self.y = None        # Save the output vector of model in this
        self.targets = None  # Save the targets in forward_pass in this variable
        self.lmd = config['L2 const']
        for i in range(len(config['layer_specs']) - 1):
            self.layers.append(Layer(config, i))

            if i < len(config['layer_specs']) - 2:
                self.layers.append(Activation(config['activation']))

    def forward_pass(self, x, targets=None):
        """
        Write the code for forward pass through all layers of the model and return loss and predictions.
        If targets == None, loss should be None. If not, then return the loss computed.
        """
        self.x = x
        self.targets = targets
        weight_loss = None if targets is None else 0
        for layer in self.layers:
            x = layer.forward_pass(x)
            if weight_loss is not None:
                try:
                    weight_loss += layer.weight_loss_func()
                except AttributeError:
                    pass 
        self.y = x
        loss = None if targets is None else self.loss_func(self.y, targets, weight_loss)
        return  loss, self.y


    def loss_func(self, logits, targets, weights):
        '''
        find cross entropy loss between logits and targets
        '''
        logits = np.sum(logits, axis=0).reshape(1,-1)
        loss = np.power(np.sum(targets - logits),2) + self.lmd*weights

        return loss

    def backward_pass(self):
        '''
        implement the backward pass for the whole network.
        hint - use previously built functions.
        '''
        
        delta = self.targets - self.y
        for layer in reversed(self.layers):
            delta = layer.backward_pass(delta)


def trainer(model, train, target, config, verbose=False):
    """
    Train the network from the architecture from above
    """
    num_epochs = config['epochs']
    loss_train = np.full(num_epochs, np.nan)
    loss_test = np.full(num_epochs, np.nan)
    err = np.full(config['batch_size'], np.nan)
    for epoch in range(config['epochs']):

        print("Epoch {}".format(epoch + 1))
        permute = np.random.permutation(train.shape[0])

        for (batch, i )in zip(np.split(permute, config['batch_size']), range(0,config['batch_size'])):
            err[i], _ = model.forward_pass(train[batch], target[batch])
            model.backward_pass()
            if(i > 0 and err[i] < err[-1]):
                model.step(accept=True)

        err = np.full(config['batch_size'], np.nan)
        loss, _ = model.forward_pass(train[batch], target[batch])
        loss_train[epoch] = loss
        # test, _ = model.forward_pass(test, test_target)
        # loss_test[epoch] = test
        if verbose:
            print("\tTraining Loss: {}\n\t".format(loss_train[epoch]))

        if not (config["early_stop"] and epoch > config["early_stop_epoch"]):
            #save the best weights
            for i in np.arange(0,len(model.layers),2):
                model.layers[i].save_weights()

    return  loss_train, loss_test

def test(model, X_test, y_test, config):
    """
    Write code to run the model on the data passed as input and return accuracy.
    """
    accuracy, _, _ = model.forward_pass(X_test, y_test)

    return accuracy


if __name__ == "__main__":
    #generate function
    y, x = generate_input(1,500)
    ty, tx = generate_input(20,500)

    print("whats going in")
    print(y.shape)
    print(x.shape)
    model = Neuralnetwork(config)
    loss, test_loss = trainer(model, x, y, config,verbose=True)
    

