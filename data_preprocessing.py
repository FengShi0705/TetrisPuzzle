# Preprocessing data txt file into numpy array
import json
import numpy as np
def preprocess(file):
    """
    preprocess file into numpy array
    :param file: txt file
    :return: [sample_size, 3(x_local,x_global, y)]
    """
    data=[]
    with open(file,'r') as f:
        for line in f:
            info = json.loads(line.strip())
            T = info['T']
            T = np.array(T,dtype=np.float32)
            S = info['S']
            processPuzzle(T,S,data)

    return data

def processPuzzle(T,S,data):
    row = T.shape[0]
    col = T.shape[1]
    for x in range(0, col):
        for y in range(0, row):
            input_data = processSquare(T,x,y)
            output_data = np.zeros(77,dtype=np.float32)
            if S[y][x][0]==0:
                output_data[76] = 1.0
            else:
                output_data[ 4*(S[y][x][0]-1) + S[y][x][1] ] = 1.0

            data.append((input_data,output_data))

    return

def processSquare(T,x,y):
    padT = np.lib.pad(T, 3, 'constant')
    x_ = x+3
    y_ = y+3
    crop = padT[y_-3:y_+4, x_-3:x_+4]
    flat = crop.reshape(49)
    return flat


def data_batch_iter(training_data, batch_size, num_epochs, shuffle=True):
    """
    generate batch data from training_data
    :param training_data:
    :param batch_size:
    :param num_epochs:
    :param shuffle:
    :return:
    """
    data = np.array(training_data)
    data_size = len(data)
    num_batches_per_epoch = int( (len(data)-1)/batch_size ) + 1

    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min( data_size, (batch_num+1)*batch_size  )
            batch_data = shuffled_data[start_index:end_index]
            x,y = zip(*batch_data)
            x = np.array(x)
            y = np.array(y)
            yield x,y


