import numpy as np
import tensorflow as tf

class Trainer:
    """
    The Trainer class handles training the tensorflow RMI model under various conditions with
    different datasets. After initializing a Trainer with a dataset and model, calling fit()
    and predict() train the model and return predictions by the model on new data, respectively.
    """
    
    def __init__(self, dataset: tf.data.Dataset, model):
        """
        Initialize a new Trainer session with the provided dataset and model.
        
        :param dataset: Dataset to train on.
        :param model: tf.keras.Model to be trained.
        """
        self.dataset = dataset
        self.model = model
        
    def fit(self, progressCb=None, **kwargs):
        """
        Fit the Trainer's model to its dataset
        
        :param progressCb: Callback for progress updates.
        :param **kwargs: Keyword Arguments to send to the model's fit() method.
        """
        if progressCb is not None:
            raise Exception("Not Implemented")
        loss = tf.keras.losses.MeanSquaredError()
        optim = tf.keras.optimizers.Adam(learning_rate=1e-4)
        self.model.compile(loss=loss, optimizer=optim)
        self.model.fit(self.dataset, **kwargs)
        
    def predict(self, new_dataset):
        """
        Predict indices for the provided new dataset.
        
        :param new_dataset: New data to predict on.
        """
        return self.model(new_dataset)