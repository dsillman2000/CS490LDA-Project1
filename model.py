import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
    
# def one_hot_argmax(x):
#     argm = tf.argmax(x[0])
#     onehot = tf.zeros_like(x)
#     onehot[argm] = 1
#     return onehot

# max_gate = keras.layers.Lambda(one_hot_argmax)

def new_stage_model(next_stage_size=1, n_layers=2, n_neurons=32):
    """
    Generates a tf.keras.Model that is an MLP with ReLU activations that
    will be wrapped in a layer and incorporated into the RMI model. Model
    is only generated, not trained.
    """
    inp = keras.Input(shape=(1,))
    x = inp
    for l in range(n_layers):
        x = keras.layers.Dense(n_neurons, activation='relu',kernel_initializer='zeros')(x)
    outp = keras.layers.Dense(next_stage_size, activation='relu',kernel_initializer='zeros')(x)
    stage_model = keras.Model(inp, outp)
    stage_model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3), loss=keras.losses.MeanSquaredError())
    return stage_model
    
def build_fit_rmi_model(dataset, stages=(1,2,3), n_layers=2, n_neurons=32):
    """
    Generates and fits the component stage models to their corresponding subsets of
    the data. See psuedocode in _Algorithm 1_ for more info.
    """
    inp = keras.Input(shape=(1,))
    M = len(stages)
    tmp_records = [[dataset]]
    index = [[]]
    for i in range(M):
        for j in range(stages[i]):
            next_stage_size = 1 if i + 1 == len(stages) else stages[i + 1]
            nn = new_stage_model(next_stage_size=1, n_layers=n_layers, n_neurons=n_neurons)
            if len(tmp_records[i][j]) > 0:
                nn.fit(tmp_records[i][j], epochs=20, verbose=0)
            if len(index) <= i:
                index.append([])
            index[i].append(nn)
            if i < M:
                xsub = [[] for _ in range(next_stage_size)]
                ysub = [[] for _ in range(next_stage_size)]
                tmp_records.append([])
                for r in tmp_records[i][j]:
                    rx, ry = r
                    pred = int(index[i][j](rx)[0] / next_stage_size)
                    pred = max(min(pred, next_stage_size-1), 0)
                    xsub[pred].append(rx)
                    ysub[pred].append(ry)
                for p in range(next_stage_size):
                    subset = tf.data.Dataset.from_tensor_slices((xsub[p], ysub[p]))
                    print(f"[i={i},j={j},p={p}]len(subset)={len(subset)}")
                    tmp_records[-1].append(subset)
    # TODO: Handle error_threshold w/ B-Trees n shiet
    return index

def predict_rmi_model(rmi, inputs):
    assert len(rmi[0]) == 1, "First stage of model must be size-1"
    M = len(rmi)
    j = 0
    for i in range(M):
        next_stage_size = 1 if i + 1 == len(rmi) else len(rmi[i + 1])
        nn = rmi[i][j]
        pred = int(nn.predict(inputs)[0] / next_stage_size)
        pred = max(min(pred, next_stage_size-1), 0)
        j = pred
    return nn.predict(inputs)