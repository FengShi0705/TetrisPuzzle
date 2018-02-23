# Preprocessing data txt file into numpy array
import json
import numpy as np


def preprocess(file,WindowSize,Half_width,Half_height,border_padValue):
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
            processPuzzle(T,S,data,WindowSize,Half_width,Half_height, border_padValue)

    return data

def processPuzzle(T,S,data,WindowSize,Half_width,Half_height, border_padValue):
    row = T.shape[0]
    col = T.shape[1]
    if S:
        for x in range(0, col):
            for y in range(0, row):
                input_data = processSquare(T,x,y,WindowSize,Half_width,Half_height,border_padValue)
                output_data = np.zeros(77,dtype=np.float32)
                if S[y][x][0]==0:
                    output_data[76] = 1.0
                else:
                    output_data[ 4*(S[y][x][0]-1) + S[y][x][1] ] = 1.0

                data.append((input_data,output_data))
    else:
        for x in range(0, col):
            for y in range(0, row):
                input_data = processSquare(T, x, y,WindowSize,Half_width,Half_height,border_padValue)
                data.append(input_data)
    return

def processSquare(T,x,y,WindowSize,Half_width,Half_height, border_padValue=0.0):
    padT = np.lib.pad(T, [[Half_height],[Half_width]], 'constant', constant_values = border_padValue)
    x_ = x+Half_width
    y_ = y+Half_height
    crop = padT[y_-Half_height:y_+Half_height+1, x_-Half_width:x_+Half_width+1]
    flat = crop.reshape(WindowSize[0]*WindowSize[1])
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



def data_batch_iter_three(training_data, batch_size, num_epochs, shuffle=True):
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
            x,y,p = zip(*batch_data)
            x = np.array(x)
            y = np.array(y)
            p = np.array(p)
            yield x,y,p


def data_batch_iter_six(training_data, batch_size, num_epochs, shuffle=True):
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
            x, action_taken, reward, discount, xend, legal_act = zip(*batch_data)
            x = np.array(x)
            action_taken = np.array(action_taken)
            reward = np.array(reward)
            discount = np.array(discount)
            xend = np.array(xend)
            legal_act = np.array(legal_act)
            yield x, action_taken, reward, discount, xend, legal_act


def data_test_three(test_data):
    test_x, test_y, test_p = zip(*test_data)
    test_x = np.array(test_x)
    test_y = np.array(test_y)
    test_p = np.array(test_p)
    return test_x, test_y, test_p

def indexto_shape_pos(i):
    """
    map index (ranging from 0 to 76) to shape and position
    :param i:
    :return:
    """
    shape = int(i/4) + 1
    pos = i % 4
    if shape == 20:
        shape=0
    return shape,pos

def shapePosto_index(shape,pos):
    """
    using shape and position to get index information
    :param shape:
    :param pos:
    :return:
    """
    index= 4*(shape-1) + pos
    if shape==0:
        index = 76
    return index



Goldenpositions = {
    1: np.array([[0,0],[1, 0], [0, 1], [1, 1]]),
    2: np.array([[0,0],[0, 1], [0, 2], [0, 3]]),
    3: np.array([[0,0],[1, 0], [2, 0], [3, 0]]),
    4: np.array([[0,0],[0, 1], [0, 2], [1, 2]]),
    5: np.array([[0,0],[-2, 1], [-1, 1], [0, 1]]),
    6: np.array([[0,0],[1, 0], [1, 1], [1, 2]]),
    7: np.array([[0,0],[1, 0], [2, 0], [0, 1]]),
    8: np.array([[0,0],[0, 1], [-1, 2], [0, 2]]),
    9: np.array([[0,0],[1, 0], [2, 0], [2, 1]]),
    10: np.array([[0,0],[1, 0], [0, 1], [0, 2]]),
    11: np.array([[0,0],[0, 1], [1, 1], [2, 1]]),
    12: np.array([[0,0],[0, 1], [1, 1], [0, 2]]),
    13: np.array([[0,0],[-1, 1], [0, 1], [1, 1]]),
    14: np.array([[0,0],[-1, 1], [0, 1], [0, 2]]),
    15: np.array([[0,0],[1, 0], [2, 0], [1, 1]]),
    16: np.array([[0,0],[1, 0], [-1, 1], [0, 1]]),
    17: np.array([[0,0],[0, 1], [1, 1], [1, 2]]),
    18: np.array([[0,0],[1, 0], [1, 1], [2, 1]]),
    19: np.array([[0,0],[-1, 1], [0, 1], [-1, 2]])
}

def one_to_threeotherpositions(x, y, shape, pos):
    """
    using one point to get all four point information of the piece
    :param x:
    :param y:
    :param shape:
    :param pos:
    :return:
    """
    info = []
    positions = []
    if shape != 0:
        for p in [0, 1, 2, 3]:
            if p != pos:
                (x_, y_) = Goldenpositions[shape][p] - Goldenpositions[shape][pos] + np.array([x, y])
                positions.append((x_, y_))
                info.append([(x_, y_), shape, p])
    return info, positions
