""" Inspired by example from
https://github.com/Vict0rSch/deep_learning/tree/master/keras/recurrent
Uses the TensorFlow backend
The basic idea is to detect anomalies in a time-series.
"""
#import matplotlib.pyplot as plt
import sys
import numpy as np
import time
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from savitzky_golay import savitzky_golay
from keras.models import load_model
np.random.seed(1234)

output = open('out2.log','w')
sys.stdout = output
# Hyper-parameters
sequence_length = 100
random_data_dup = 10  # each sample randomly duplicated between 0 and 9 times, see dropin function
#epochs = 2
batch_size = 512
class_size = 30000

def dropin(X, y):

    """ The name suggests the inverse of dropout, i.e. adding more samples. See Data Augmentation section at
    http://simaaron.github.io/Estimating-rainfall-from-weather-radar-readings-using-recurrent-neural-networks/
    :param X: Each row is a training sequence
    :param y: Tne target we train and will later predict
    :return: new augmented X, y
    """
    print("X shape:", X.shape)
    print("y shape:",y.shape)
    X_hat = []
    y_hat = []
    for i in range(0, len(X)):
        for j in range(0, np.random.random_integers(0,20)):
            X_hat.append(X[i, :])
            y_hat.append(y[i])
    return np.asarray(X_hat), np.asarray(y_hat)


def z_norm(result):
    result_mean = result.mean()
    result_std = result.std()
    result -= result_mean
    result /= result_std
    return result, result_mean

def signal_concat(signal, signal_add):
    if signal.shape[0] > class_size: return signal
    if signal is None:
        signal = signal_add
    else:
        signal = np.concatenate((signal, signal_add))
    return signal


def ladd(ll, add_l, labid):
    if len(ll) > class_size: return ll
    for i in range(0, add_l.shape[0] - sequence_length):
        tmp = add_l[i:i + sequence_length].tolist()
        tmp.append(labid)
        ll.append(tmp)
    return ll

def get_split_prep_data(train_ratio):
    file = [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 111, 112, 113, 114, 115, 116, 117, 118, 119, 121, 122,
            123, 124,200, 201, 202, 203, 205, 207, 208, 209, 210, 212, 213, 214, 215, 217, 219, 220, 221, 222, 223,
            228, 230,231, 232, 233, 234]
    file = [str(i) for i in file]
    normal = [];fusion = [];atrial = [];ventri = []
    for f in file:

        nor = np.load('mitdb/N_' + f + '.npy')
        atr = np.load('mitdb/A_' + f + '.npy')
        fus = np.load('mitdb/F_' + f + '.npy')
        ven = np.load('mitdb/V_' + f + '.npy')

        np.reshape(nor, (nor.shape[0], 2))
        np.reshape(fus, (fus.shape[0], 2))
        np.reshape(atr, (atr.shape[0], 2))
        np.reshape(ven, (ven.shape[0], 2))

        print 'Ori_Shape:', nor.shape, fus.shape, atr.shape, ven.shape

        if nor.shape[0] > 0:
            nor[:, 0] = savitzky_golay(nor[:, 0], 11, 3)
            nor[:, 1] = savitzky_golay(nor[:, 1], 11, 3)
        if atr.shape[0] > 0:
            atr[:, 0] = savitzky_golay(atr[:, 0], 11, 3)
            atr[:, 1] = savitzky_golay(atr[:, 1], 11, 3)
        if fus.shape[0] > 0:
            fus[:, 0] = savitzky_golay(fus[:, 0], 11, 3)
            fus[:, 1] = savitzky_golay(fus[:, 1], 11, 3)
        if ven.shape[0] > 0:
            ven[:, 0] = savitzky_golay(ven[:, 0], 11, 3)
            ven[:, 1] = savitzky_golay(ven[:, 1], 11, 3)

        normal = ladd(normal, nor, [1,0,0,0])
        atrial = ladd(atrial, atr, [0,1,0,0])
        fusion = ladd(fusion, fus, [0,0,1,0])
        ventri = ladd(ventri, ven, [0,0,0,1])

    normal = np.array(normal)[:class_size]
    atrial = np.array(atrial)[:class_size]
    fusion = np.array(fusion)[:class_size]
    ventri = np.array(ventri)[:class_size]


    normal_size = len(normal)
    fusion_size = len(fusion)
    atrial_size = len(atrial)
    ventri_size = len(ventri)

    print 'Normal_size', normal_size
    print 'Fusion_size', fusion_size
    print 'Atrial_size', atrial_size
    print 'Ventri_size', ventri_size

    result = np.concatenate((normal, atrial, fusion, ventri))
    np.random.shuffle(result)


    label = np.array(result[:, -1].tolist())
    data = np.array(result[:,:-1].tolist())
    result = data
    print('Result shape', result.shape, label.shape)
    result, result_mean = z_norm(result)
    print('Result, Result_mean', result.shape, result_mean)
    print "Mean of train data : ", result_mean
    print "Train data shape  : ", result.shape


    # label = []
    # idx = 0
    # while idx < normal_size:
    #     label.append([1,0,0,0])
    #     idx += 1
    # idx = 0
    # while idx < atrial_size:
    #     label.append([0,1,0,0])
    #     idx += 1
    # idx = 0
    # while idx < fusion_size:
    #     label.append([0,0,1,0])
    #     idx += 1
    # idx = 0
    # while idx < ventri_size:
    #     label.append([0,0,0,1])
    #     idx += 1

    print 'Resule Label Shape', result.shape, label.shape

    train_size = int(result.shape[0] * train_ratio)
    X_train, y_train = result[:train_size], label[:train_size]
    X_test, y_test = result[train_size:], label[train_size:]

    print 'X_shape', X_train.shape, X_test.shape
    print 'y_shape', y_train.shape, y_test.shape
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 2))
    X_test = np.reshape(X_test, (X_test.shape[0], X_train.shape[1], 2))

    return X_train, y_train, X_test, y_test

def build_model():
    model = Sequential()
    layers = {'input': 2, 'hidden1': 64, 'hidden2': 128, 'hidden3': 64, 'hidden4':32, 'output': 4}

    model.add(LSTM(
            input_length=sequence_length,
            input_dim=layers['input'],
            output_dim=layers['hidden1'],
            return_sequences=True))

    model.add(Dropout(0.2))

    model.add(LSTM(
            layers['hidden2'],
            return_sequences=True,
            activation='tanh'))

    model.add(Dropout(0.2))

    model.add(LSTM(
            layers['hidden3'],
            return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(
        layers['hidden4'],
        return_sequences=False))
    model.add(Dropout(0.2))

    model.add(Dense(
            output_dim=layers['output'],
            activation="softmax"))

    start = time.time()
    model.compile(loss="categorical_crossentropy", optimizer="sgd", class_mode='categorical',metrics=['accuracy'])
    print "Compilation Time : ", time.time() - start
    return model

def run_network(model=None, begin_epoch = 0, num_epoch = 100):
    print 'Begin_Epoch:', begin_epoch, 'Num_Epoch:', num_epoch
    global_start_time = time.time()
    #epochs = 100

    print 'Loading data... '
    X_train, y_train, X_test, y_test = get_split_prep_data(0.8)

    print '\nData Loaded. Compiling...\n'

    if model is None:
        model = build_model()

    try:
        print("Training")
        now_epoch = 0
        if begin_epoch != 0:
            model = load_model('model/model' + str(begin_epoch - 1) + '.h5')
        while now_epoch < num_epoch:
            print('Epoch:', now_epoch)
            model.fit(
                    X_train, y_train,
                    batch_size=batch_size, nb_epoch=1, validation_split=0.05, verbose=2)
            model.save('model/model' + str(now_epoch) + '.h5')
            now_epoch += 1
            predicted = model.predict(X_test)
            pre_label = [np.argmax(l) for l in predicted]
            ori_label = [np.argmax(l) for l in y_test]
            num = sum([1 if (pre_label[i] == ori_label[i]) else 0 for i in range(len(pre_label))])
            print 'Accuracy:', num * 1.0 / len(pre_label)
        model = load_model('model/model' + str(now_epoch-1) + '.h5')
        #print("Predicting")
        #predicted = model.predict(X_test)
        #print  predicted
        #print("shape of predicted", np.shape(predicted), "size", predicted.size)
        #print("Reshaping predicted")
        #predicted = np.reshape(predicted, (predicted.size,))
        #print  predicted
    except KeyboardInterrupt:
        print("prediction exception")
        print 'Training duration (s) : ', time.time() - global_start_time
        return model, y_test, 0

    model = load_model('model/model' + str(now_epoch - 1) + '.h5')
    print 'Training duration (s) : ', time.time() - global_start_time
    score, acc = model.evaluate(X_test, y_test)
    print score, acc
    print type(score), type(acc)
    return score, acc

if __name__ == '__main__':
    run_network(begin_epoch = 13, num_epoch = 1000)
    # # a = np.array([[1,2],[3,4],[1,1]])
    # # b = np.array([[5,6],[7,8],[2,2]])
    # # c = np.array([6,6,6])
    # # #c = np.reshape(c, (len(c), 1))
    # # print a.shape, b.shape, c.shape
    # # print np.column_stack((a,c))
    # path_to_dataset = '/home/lcc/ECG/'
    # normal = np.load(path_to_dataset + 'normal.npy')
    # anomal = np.load(path_to_dataset + 'anomal.npy')
    # print normal.shape, anomal.shape
    # label = np.array([0] * normal.shape[0] + [1] * anomal.shape[0])
    # data = np.row_stack((normal,anomal))
    # data = np.column_stack((data, label))
    # np.save('data.npy',data)
