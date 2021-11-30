import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import tqdm
import pickle

BATCHSIZE = 64

class MaxAbsoluteError(keras.metrics.Metric):
    def __init__(self, name='max_absolute_error', **kwargs):
        super(MaxAbsoluteError, self).__init__(name=name, **kwargs)
        self.maxae = self.add_weight(name='mae', initializer='zeros', dtype='float32')
    def update_state(self, y_true, y_pred, sample_weight=None):
        values = tf.math.abs(tf.subtract(y_true, y_pred))
        self.maxae.assign(tf.math.maximum(tf.reduce_max(values), tf.reduce_max(self.maxae.value())))
    def result(self):
        return self.maxae
    
class TQDMCallback(keras.callbacks.Callback):
    def __init__(self, tqdmiter, nepochs, sn, mn):
        self.tqdmiter = tqdmiter
        self.nepochs = nepochs
        self.tqdmiter.total = (self.nepochs)
        self.base = f"(S{sn}, M{mn})"
    def on_epoch_end(self, epoch, logs=None):
        keys = list(logs.keys())
        self.tqdmiter.set_description(f"{self.base}  Finished Epoch #{epoch + 1}  MaxAE {logs.get('max_absolute_error',0.0)} Mean AE {logs.get('mean_absolute_error',0.0)}")
        self.tqdmiter.n = epoch + 1

def bootstrap_pad(vector, iaxis_pad_width, iaxis, kwargs):
    center = list(vector[iaxis_pad_width[0]:-iaxis_pad_width[1]])
    seed = kwargs.get("seed", None)
    if seed:
        np.random.seed(seed)
    if iaxis_pad_width[0] > 0:
        boot = np.random.choice(center, size=(iaxis_pad_width[0]))
        vector[:iaxis_pad_width[0]] = boot
    if iaxis_pad_width[1] > 0:
        boot = np.random.choice(center, size=(iaxis_pad_width[1]))
        vector[-iaxis_pad_width[1]:] = boot
        
def new_stage_model(next_stage_size=1, n_layers=2, n_neurons=32, initializer='identity', **kwargs):
    """
    Generates a tf.keras.Model that is an MLP with ReLU activations that
    will be wrapped in a layer and incorporated into the RMI model. Model
    is only generated, not trained.
    """
    inp = keras.Input(shape=(1,1))#, batch_size=BATCHSIZE)
    x = inp
    for l in range(n_layers):
        x = keras.layers.Dense(n_neurons, activation='relu',kernel_initializer=initializer)(x)
    outp = keras.layers.Dense(next_stage_size, activation='relu',kernel_initializer=initializer)(x)
    stage_model = keras.Model(inp, outp)
    stage_model.compile(**kwargs)
    return stage_model
    
def zero_model(y, **kwargs):
    inp = keras.Input(shape=(1,1))
    x = inp
    outp = keras.layers.Dense(1, kernel_initializer='zeros', bias_initializer=keras.initializers.Constant(y), trainable=False)(x)
    model = keras.Model(inp, outp)
    model.compile(**kwargs)
    return model
    
def build_fit_rmi_model(dataset, stages=(1,3,4), n_layers=2, n_neurons=32, n_epochs=10, verbose='auto', initializer='identity', **kwargs):
    """
    Generates and fits the component stage models to their corresponding subsets of
    the data. See psuedocode in _Algorithm 1_ for more info.
    """
    maxind = max(list(dataset.shape))
    inp = keras.Input(shape=(1,))
    M = len(stages)
    tmp_records = [[dataset]]
    index = [[]]
    progress = tqdm.tqdm(range(M))
    print(f"FITTING RMI MODEL WITH BATCH SIZE {BATCHSIZE}")
    for i in progress:
        next_stage_size = 1 if i + 1 == len(stages) else stages[i + 1]
        if i < M - 1:
                tmp_records.append([[] for _ in range(next_stage_size)])
        for j in range(stages[i]):
            if len(tmp_records[i][j][0]) == 0:
                true_y = tmp_records[i][j-1][1][-1]
                nn = zero_model(true_y)
            elif len(tmp_records[i][j][0]) <= 5:
                true_y = np.mean(tmp_records[i][j][1])
                nn = zero_model(true_y)
            else :
                basicargs = {"next_stage_size":1, "n_layers":n_layers, "n_neurons": n_neurons, "initializer":initializer}
                nn = new_stage_model(**basicargs, **kwargs)
#                 if i > 0:
#                     nn.load_weights('rootmodel.tf')
                if len(tmp_records[i][j][0]) % BATCHSIZE != 0:
                    tmp_records[i][j] = np.array(tmp_records[i][j]).reshape(2,-1,1)
                    nextup = (len(tmp_records[i][j][0])//BATCHSIZE + 1) * BATCHSIZE
                    tmp_records[i][j] = np.pad(tmp_records[i][j], ((0,0),(0,nextup-len(tmp_records[i][j][0])),(0,0)), mode=bootstrap_pad, kwargs={"seed":i}).reshape(2,-1,1)
                    sh = tmp_records[i][j].shape
                    assert len(sh) == 3 and sh[0] == 2 and sh[2] == 1
                hist = nn.fit(tmp_records[i][j][0], tmp_records[i][j][1], epochs=n_epochs, verbose=verbose, callbacks=[TQDMCallback(progress, n_epochs, i+1, j+1)], batch_size=BATCHSIZE)#0)
                if i == 0:
                    nn.save_weights("rootmodel.tf")
                    with open("submodel.arch", "wb") as f:
                        pickle.dump(basicargs, f)
            if len(index) <= i:
                index.append([])
            index[i].append(nn)
            if i < M - 1:
                pred = index[i][j].predict(tmp_records[i][j][0], batch_size=BATCHSIZE).reshape(-1) / maxind * next_stage_size
                pred = pred.astype(int)
                pred[pred < 0] = 0
                pred[pred >= next_stage_size] = next_stage_size-1
                tmp_all = tmp_records[i][j]
                for p in range(next_stage_size):
                    tmp = tmp_all[:,pred==p,:]
                    if len(tmp) == 0:
                        continue
                    subset = tmp.reshape(2,-1,1)
                    if verbose: 
                        print(f"[i={i},j={j},p={p}]len(subset)={subset.shape}")
                    tmp_records[-1][p].append(subset)
        # Join subsets within each bucket
        for p in range(next_stage_size):
            if not type(tmp_records[-1][p]) == list:
                tmp_records[-1][p] = [tmp_records[-1][p]]
            tmp_records[-1][p] = np.concatenate(tmp_records[-1][p], axis=1)
            assert type(tmp_records[-1][p]) == np.ndarray, "type mismatch 2!"
    return index

def predict_rmi_model(rmi, inputs, maxind=None, with_leaf=False):
    assert len(rmi[0]) == 1, "First stage of model must be size-1"

    if maxind is None:
        maxind = len(inputs[0])
    M = len(rmi)
    j = [0] * len(inputs[0])
    leaf_idx = 0
    nns = []
    for i in range(M):
        next_stage_size = 1 if i + 1 == len(rmi) else len(rmi[i + 1])
        uniqj = np.unique(j)
        nns = [rmi[i][uniqj[k]] for k in range(len(uniqj))]
        pred = {}
        for uj in range(len(uniqj)):
            uinputs = inputs[0,j==uniqj[uj],:]
            pred[uniqj[uj]] = (nns[uj].predict(uinputs, batch_size=1).reshape(-1) / maxind * next_stage_size).astype(int)
        pred = np.concatenate(list(pred.values()))
        if i == M - 1:
            leaf_idx = np.array(j).reshape(-1)
            pred = {}
            for uj in range(len(uniqj)):
                uinputs = inputs[0,j==uniqj[uj],:]
                pred[uniqj[uj]] = (nns[uj].predict(uinputs, batch_size=1).reshape(-1))
            pred = np.concatenate(list(pred.values()))
            pred[pred<0] = 0
            pred[pred>maxind] = maxind
        else:
            pred = [min(max(pred[i], 0), next_stage_size-1) for i in range(len(inputs[0]))]
        j = pred
    if with_leaf:
        return pred, leaf_idx
    return pred