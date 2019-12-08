import numpy as np 
import matplotlib.pyplot as plt

config = {}
config['layer_specs'] = [3, 3 ,1]
config['activation'] = 'tanh' 
config['epochs'] = 5 # Number of epochs to train the model
config['early_stop'] = True  # Implement early stopping or not
config['early_stop_epoch'] = 50  # Number of epochs for which validation loss increases to be counted as overfitting
config['batch_size'] = 1
# LM => stands fpr 'Levenberg_Marquardt' algorithm
config['optimizer'] = 'LM'
config['gamma'] = 0.2
config['L2 const'] = 0.001



def generate_input(BOUND,UBOUND, NUM_SAMPLE, select_func):
    
    x1 = np.linspace(BOUND,UBOUND,NUM_SAMPLE, dtype=float)
    x2 = np.linspace(BOUND,UBOUND,NUM_SAMPLE, dtype=float)
    x3 = np.linspace(BOUND,UBOUND,NUM_SAMPLE, dtype=float)
    x1 = x1[np.random.permutation(x1.shape[0])].reshape(1,NUM_SAMPLE)
    x2 = x2[np.random.permutation(x2.shape[0])].reshape(1,NUM_SAMPLE)
    x3 = x3[np.random.permutation(x3.shape[0])].reshape(1,NUM_SAMPLE)
    
    """
        First function is g(x) = x1*x2 + x3
    """
    if select_func == 1:
        target = np.multiply(x1,x2)+x3
        train = np.vstack((np.vstack((x1,x2)),x3))
    """
        Second function is g(x) = x1*x2 + x1*x3 >> it has two nonlinear relation ships
    """
    if select_func == 2:
        target = np.multiply(x1,x2)+x1*x3
        train = np.vstack((np.vstack((x1,x2)),x3))

    return (target.T , train.T)

class Activation:
    def __init__(self, activation_type = "tanh"):
        self.activation_type = activation_type
        self.x = None # Save the input 'x' for sigmoid or tanh or ReLU to this variable since it will be used later for computing gradients.

    def forward_pass(self, a):
        
        return self.tanh(a)

    def backward_pass(self, delta,step):
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
        self.selelct_optizmizer = 'GD'
        self.w = 1 / (in_units + out_units) * np.random.randn(in_units, out_units)
        self.step = True
        self.b = np.zeros((1, out_units)).astype(np.float32)    # Bias
        self.temp_w = None
        self.temp_b = self.b
        self.gamma = config['gamma']
        self.learning_rate = 0.001
        self.x = None    # Save the input to forward_pass in this
        self.a = None    # Save the output of forward pass in this (without activation)
        self.d_x = None    # Save the gradient w.r.t x in this
        self.d_w = None    # Save the gradient w.r.t w in this
        self.d_b = None    # Save the gradient w.r.t b in this
        self.step = False

    def forward_pass(self, x):
        """
        Write the code for forward pass through a layer
        """

        self.x = x
        self.a = x @ self.w + self.b
        # print('shape of x'+ str(x.shape))
        # print('shape of weight'+ str(self.w.shape))
        print("weight")
        print(self.w)
        print("Biase")
        print(self.b)

        return self.a

    def backward_pass(self, delta, step):
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
        if step:
            I1 = np.eye(self.w.shape[0]) #creating the dynamic shape I matrix
            #I2 = np.eye(self.b.shape[0])

            """
            after delta has been passed back,
            implementing Levenberg Marquardt optimizer to compute the new weights going out of this layer (lamda varies, adaptive learning)
            wt+1 for this hidden layer:
            """
            if self.selelct_optizmizer == 'LM':
                self.w = self.w - np.linalg.inv(self.d_w.T @ self.d_w + self.gamma*I1) @ self.d_w @ self.d_x
                self.temp_b = self.b - np.linalg.inv(self.d_b.T @ self.d_b + self.gamma*I2) @ self.d_b @ self.d_x
                self.w = self.temp_w  
                self.b = self.temp_b
                self.gamma = 0.08*self.gamma
                print("Accept True0"+str(self.gamma))

            if self.selelct_optizmizer == 'GD':
                self.w += self.learning_rate * self.d_w
                self.b += self.learning_rate * self.d_b

        elif not step:
            #weights dont change only gamma is stepped up
            self.gamma = 10*self.gamma
            print("Accept False"+str(self.gamma))


        return new_delta

    def weight_loss_func(self):
        """
        L2 loss function for weights
        """
        return np.linalg.norm(self.w,ord=2)**2
    


    """
        These two functions will save the best best weights of the network when early stop is occurs
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
        self.step = True
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
        find  loss between logits and targets with L2 Regulrization 
        '''
        logits = np.sum(logits, axis=0).reshape(1,-1)
        loss = np.power(np.sum(targets - logits),2) + self.lmd*weights

        return loss

    def backward_pass(self, step):
        '''
        implementing the backward pass for the whole network, all the gradients are created here
        '''
        #Step = Step
        delta = self.targets - self.y
        
        for layer in reversed(self.layers):
            delta = layer.backward_pass(delta, step)


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
           
            if(i >1 and err[i] > err[-1]):
                #print("accept")
                model.backward_pass(step=False)
            else:
                #print("dontaccept")
                model.backward_pass(step=True)

        err = np.full(config['batch_size'], np.nan)
        loss, _ = model.forward_pass(train[batch], target[batch])
        loss_train[epoch] = loss
        # test, _ = model.forward_pass(test, test_target)
        # loss_test[epoch] = test
        if verbose:
            print("\tTraining Loss: {}\n\t".format(loss_train[epoch]))

        #Early Stop Criteria
        if ( epoch > config["early_stop_epoch"] and loss > np.mean(loss_train[:-5])):
            print("Early stop accured")
            break

        if not (config["early_stop"] and epoch > config["early_stop_epoch"]):
            #save the best weights
            for i in np.arange(0,len(model.layers),2):
                model.layers[i].save_weights()

    return  loss_train, loss_test

def plot_it(train, test, graph_type, epochs, layers, activation, id):
    title = "{} Over {} Epochs with {} Hidden Layers\nUsing the {} Activation Function".format(graph_type, epochs, layers, activation)
    filename = graph_type
    if id is not None:
        filename = "{}-{}".format(id, filename)
    plt.plot(train, label='Test Loss{}'.format(graph_type))
    #plt.plot(test, label='Test {}'.format(graph_type))
    plt.ylabel(graph_type)
    plt.xlabel('Epoch')
    plt.xlim(0, epochs - 1)
    plt.title(title)
    plt.grid()
    plt.savefig(filename)
    plt.clf()


if __name__ == "__main__":
    #generate function
    y, x = generate_input(1,-1,500,2)
    #generating test set
    ty, tx = generate_input(2,-2,500,2)
    print("whats going in")
    print(y.shape)
    print(x.shape)
    model = Neuralnetwork(config)
    loss, test_loss = trainer(model, x, y, config,verbose=True)

    plot_it(loss, test_loss,"gamma", config['epochs'], 1, "tanh","02")
    


