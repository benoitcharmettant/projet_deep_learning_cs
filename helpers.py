from os.path import join

from numpy import array, zeros, nan_to_num
from pandas import read_csv


def makePredictions(model, start_window, data_description, prediction_window_length, e_params):
    # Cette fonction devient un peu une usine à gaz avec les données externes, comme j'ai un peu la flemme de tout commentez demandez moi si vous avez un problème, normalement son utilisation devrait être simple...
    
    for v in e_params.values():
        assert len(v) == prediction_window_length
    
    w = len(start_window)
    dim = start_window.shape[1]
 
    
    keys = list(data_description.keys())
    
    e_p = []
    for k in keys:
        if not data_description[k]:
            e_p.append(e_params[k])
            
    e_p = array(e_p).T
        
    
    predictable = list(data_description.values())
    non_predictable = [not elem for elem in predictable]


    predictions = zeros((w + prediction_window_length, dim))
    predictions[0:w] = start_window.reshape(w, dim)

    for i in range(prediction_window_length):
        predictions[w+i, predictable] = model.predict(predictions[i:i+w].reshape(1,w,dim))[0]
        
        if w+i < prediction_window_length and len(e_p) > 0:
            predictions[w+i, non_predictable] = e_p[w+i]
        
    p = {}
        
    for i, k in enumerate(keys):
        if data_description[k]:
            p[k] = predictions.T[i, w:]
        
    return p

def makeForcedPredictions(model, d, window_span, data_description):
    
    dim = len(d.keys())
    
    predictable = list(data_description.values())
    dim_out = sum(predictable)
    
    
    serie = array(list(d.values())).T
    
    w = window_span
    l = len(serie)

    predictions = zeros((l, dim_out))
    predictions[0:w] = serie[0:w, predictable]

    for i in range(l - w):
        predictions[w+i] = model.predict(serie[i:i+w].reshape(1,w,dim))
        
    p = {}
    
    i=0
    for k in list(d.keys()):
        if data_description[k]:
            p[k] = predictions.T[i]
            i+=1
        
    return p


def loadModel(name, dim, step, loss):
    
    import keras
    from keras.models import Sequential
    from keras.layers import Dense, SimpleRNN, LSTM
    
    model = Sequential()
    
    if name == "simpleRNN":
        model.add(SimpleRNN(units=16, input_shape=(step, dim[0]), activation="relu"))
        # model.add(Dense(8, activation="relu")) 
        model.add(Dense(dim[1], activation="relu"))
        model.compile(loss=loss, optimizer='rmsprop')
    elif name == "LSTM":
        model.add(LSTM(16, input_shape=(step, dim[0])))
        model.add(Dense(dim[1]))
        model.compile(loss=loss, optimizer='rmsprop')
    elif name == "BiLSTM":
        model.add(Bidirectional(LSTM(16), input_shape=(step, dim), merge_mode='sum'))
        model.add(Dense(dim, activation="relu"))
        model.compile(loss=loss, optimizer='rmsprop')
    elif name == "GRU":
        model.add(GRU(16, input_shape=(step, dim)))
        model.add(Dense(dim))
        model.compile(loss=loss, optimizer='rmsprop')  
    model.summary()
    
    return model

def formatData(list_dict, step, desc):
    #Les données doivent arriver au bon format et être de même dimension
    
    nb_country = len(list_dict)
    
    _ = list(list_dict[0].values())
    predictible = list(desc.values())
    
    l = len(_[0])
    n_in = len(_)
    n_out = sum(predictible)
    
    X = zeros(((l-step)*nb_country, step, n_in))
    Y = zeros(((l-step)*nb_country, n_out))
    
    for n, data_dict in enumerate(list_dict):
        
        values = array(list(data_dict.values()))

        for i in range(l - step):
            for j in range(n_in):
                X[n*(l - step) + i,:,j] = values[j, i:i+step]

            Y[n*(l - step) + i] = values[predictible, i+step]

        return X, Y

def loadOwid(path, country, data_desctription, window=(0,-1)):
    
    keys = list(data_desctription.keys())
    
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
    
    
    
    