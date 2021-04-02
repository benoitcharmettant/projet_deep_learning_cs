from os.path import join

from numpy import array, zeros, nan_to_num
from pandas import read_csv


def makePredictions(model, start_window, keys, prediction_window_length):
    w = len(start_window)
    dim = start_window.shape[1]

    predictions = zeros((w + prediction_window_length, dim))
    predictions[0:w] = start_window.reshape(w, dim)

    for i in range(prediction_window_length):
        predictions[w+i] = model.predict(predictions[i:i+w].reshape(1,w,dim))[0]

    p = {}
        
    for i, k in enumerate(keys):
        p[k] = predictions.T[i, w:]
        
    return p

def makeForcedPredictions(model, d, window_span):
    
    dim = len(d.keys())
    
    
    serie = array(list(d.values())).T
    
    w = window_span
    l = len(serie)

    predictions = zeros((l, dim))
    predictions[0:w] = serie[0:w]

    for i in range(l - w):
        predictions[w+i] = model.predict(serie[i:i+w].reshape(1,w,dim))
        
    p = {}
        
    for i, k in enumerate(list(d.keys())):
        p[k] = predictions.T[i]
        
    return p


def loadModel(name, dim, step, loss):
    
    import keras
    from keras.models import Sequential
    from keras.layers import Dense, SimpleRNN, LSTM
    
    model = Sequential()
    
    if name == "simpleRNN":
        model.add(SimpleRNN(units=16, input_shape=(step, dim), activation="relu"))
        # model.add(Dense(8, activation="relu")) 
        model.add(Dense(dim, activation="relu"))
        model.compile(loss=loss, optimizer='rmsprop')
    elif name == "LSTM":
        model.add(LSTM(16, input_shape=(step, dim)))
        model.add(Dense(dim))
        model.compile(loss=loss, optimizer='rmsprop')
    elif name == "VAE":
        model.compile(loss=loss, optimizer='rmsprop')
    model.summary()
    
    return model

def formatData(data_dict, step):
    #Les données doivent arriver au bon format et être de même dimension
    
    values = array(list(data_dict.values()))
    
    l = len(values[0])
    n = len(values)
    
    X = zeros((l-step, step, n))
    Y = zeros((l-step, n))
    
    for i in range(l - step):
        for j in range(n):
            X[i,:,j] = values[j, i:i+step]
        Y[i] = values[:, i+step]
    
    return X, Y

def loadOwid(path, country, keys, window=(0,-1)):
    
    csv = read_csv(join(path, "owid.csv"))
    
    csv_country = csv[csv["location"] == country]
    
    d = {}
    
    for k in keys:
        d[k] = nan_to_num(array(csv_country[k]), 0).astype(int)[window[0]:window[1]]
        
    return d

def processData(serie, div=1, average=False, offset=0):
    
    from numpy import mean, zeros
    
    l = len(serie)
    
    serie = serie/div
    
    if average:
        output = zeros(l-7)
        for i in range(l-7):
            output[i] = mean(serie[i:i+7])
        serie = output
        
    return serie[offset:]

def splitTrainTest(d, test_length):
    d_train, d_test = {}, {}
    
    for k in d.keys():
        d_train[k] = d[k][:-test_length]
        d_test[k] = d[k][-test_length:]
        
    return d_train, d_test
    
    
    
    