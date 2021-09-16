import tensorflow as tf
import tensorflow.keras as keras

class StageModel(keras.Model):
    def __init__(self, n_layers=2, n_neurons=32, n_children=1):
        """
        Initialize a new "building block" stage model with the specified
        numer of layers and neurons.
        
        :param n_layers: Number of layers in the stage model.
        :param n_neurons: Number of neurons per layer in the stage model.
        :param n_children: The number of stage models that descend from this stage model.
                           1 means it is a leaf model.
        """
        super(StageModel, self).__init__()
        self.n_layers = n_layers
        self.n_neurons = n_neurons
        self.n_children = n_children
        self.lays = [keras.layers.Input(shape=1)]
        for i in range(self.n_layers):
            self.lays.append(keras.layers.Dense(self.n_neurons, activation='relu'))
        self.lays.append(keras.layers.Dense(self.n_children, activation='relu'))
    
    def call(self, inputs, training=False):
        """
        Forward-pass through the stage model MLP.
        """
        # x = self.lays[0](inputs, training=training)
        x = inputs
        for i in range(1, len(self.lays)):
            x = self.lays[i](x, training=training)
        return x
    
class RecursiveModelIndex(keras.Model):
    def __init__(self, stages=(1,2,3), n_layers=2, n_neurons=32):
        """
        Initialize an RMI with the stage sequence provided in `stages`, each stage model
        containing `n_layers` layers with `n_neurons` neurons per layer.
        """
        super(RecursiveModelIndex, self).__init__()
        self.stages = stages
        self.n_layers = n_layers
        self.n_neurons = n_neurons
        self.key_input = keras.layers.Input(shape=(1,))
        self.submodels = [[] for _ in range(len(stages))]
        num_models = sum(list(self.stages))
        for i in range(len(self.stages)):
            nchild = 1 if (i+1) >= len(self.stages) else self.stages[i+1]
            stagemodels = [StageModel(n_layers=self.n_layers, n_neurons=self.n_neurons, n_children=nchild)]
            self.submodels[i] = stagemodels
        self.idx_pred = keras.layers.Maximum()
            
    def call(self, inputs, training=False):
        """
        Forward-pass through the RMI.
        """
        x = inputs
        chosen = self.submodels[0][0]
        for i in range(len(self.stages) - 1):
            confidences = chosen.call(x, training=training)[0]
            print(f"confidences={confidences}")
            chosen = self.submodels[i + 1][tf.math.argmax(confidences)]
            
        x = chosen.call(x, training=training)        
        return x