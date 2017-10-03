import tensorflow as tf
from data_preprocessing import processPuzzle,indexto_shape_pos,shapePosto_index,one_to_threeotherpositions
import numpy as np
from heapq import heappush, heappop

def calc_prob(batch_x):
    # build graph
    tf.reset_default_graph()
    x = tf.placeholder(tf.float32, shape=[None, 49])
    #y_ = tf.placeholder(tf.float32, shape=[None, 77])

    def weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def conv2d(x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def max_pool_2x2(x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding='VALID')

    # first conv layer
    W_conv1 = weight_variable([4, 4, 1, 100])
    b_conv1 = bias_variable([100])
    x_puzzle = tf.reshape(x, [-1, 7, 7, 1])

    h_conv1 = tf.nn.relu(conv2d(x_puzzle, W_conv1) + b_conv1)  # output_dim = ceil( input_dim / stride ) = 7/1 = 7

    h_pool1 = max_pool_2x2(
        h_conv1)  # output_dim = ceil( (input_dim - (filter_dim -1) ) / stride ) = ceil( (7-(2-1)) / 1 ) = 6

    # second conv layer
    W_conv2 = weight_variable([4, 4, 100, 200])
    b_conv2 = bias_variable([200])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)  # output_dim = ceil( input_dim /stride   ) = 6 / 1 = 6
    h_pool2 = max_pool_2x2(
        h_conv2)  # output_dim = ceil ( ( input_dim- (filter-1) ) / stride ) = ceil( (6 - (2-1)) / 1 ) = 5

    # densely connected layer
    W_fc1 = weight_variable([5 * 5 * 200, 2000])
    b_fc1 = bias_variable([2000])
    h_pool2_flat = tf.reshape(h_pool2, [-1, 5 * 5 * 200])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    # dropout layer
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # classifier layer
    W_fc2 = weight_variable([2000, 77])
    b_fc2 = weight_variable([77])
    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    # softmax and cross_entropy
    prob = tf.nn.softmax(y_conv)

    # save
    saver = tf.train.Saver()

    with tf.Session() as sess:
        # Restore variables from disk.
        saver.restore(sess, "../TetrisPuzzle_saver/best_test_model.ckpt")
        print("Model restored.")
        probability = prob.eval(feed_dict={
            x:batch_x, keep_prob:1.0
        })

    return probability


def Tetris(T):
    #preprocess data
    T = np.array(T, dtype=np.float32)
    inputdata=[]
    processPuzzle(T,None,inputdata)
    inputdata = np.array(inputdata)

    #probability
    all_probability = calc_prob(inputdata)

    M,S = fill(all_probability, T.shape)

    return M,S

def fill(all_probability, shape):
    row = shape[0]
    col = shape[1]
    M = np.empty([row, col], object)
    S = np.empty([row,col], object)
    Node_list =[]
    NodeC.Node_matrix = np.empty([row,col], dtype=object)
    NodeC.Shape = (row,col)

    # create node objects and save into Node_matrix
    Pnum = 0
    c_nd = np.arange(0, row * col, 1)
    np.random.shuffle(c_nd)
    for x in range(0,col):
        for y in range(0,row):
            node = NodeC( all_probability[Pnum], x, y, c_nd[Pnum] )
            Pnum += 1
            NodeC.Node_matrix[y][x] = node

    #calculate initial score and pop into Node_list
    for rnodes in NodeC.Node_matrix:
        for node in rnodes:
            node.update_score()
            heappush(Node_list, (-node.score, node.count, node))


    # fill
    Pid = 1
    while len(Node_list) > 0:
        (neg_score, c_, node) = heappop(Node_list)
        if check_and_update(node, M, S, Pid, Node_list):
            if node.shapeid !=0 :
                Pid += 1
            print('fill node: x({}), y({}), shape({}), pos({}), and the other corresponding three nodes'.format(node.x, node.y, node.shapeid, node.pos))
        else:
            node.update_score()
            heappush(Node_list, (-node.score, node.count, node))

    return M,S

class NodeC(object):
    Node_matrix = None
    Shape = None

    def __init__(self, probability, x, y, count):
        self.count = count
        self.probability=probability
        self.prob_heap=[]

        rd_class = np.arange(0, 77, 1)
        np.random.shuffle(rd_class)
        for i, p in enumerate(probability):
            shapeid = int(i/4) + 1
            pos = i % 4
            if shapeid==20:
                shapeid = 0
            heappush(self.prob_heap, (-p, rd_class[i], ( shapeid, pos ) ))

        self.x=x
        self.y=y

    def update_score(self):
        (neg_p, rdC_, (shapeid,pos))=heappop(self.prob_heap)
        p = -neg_p
        self.shapeid=shapeid
        self.pos=pos
        info, positions = one_to_threeotherpositions(self.x, self.y, self.shapeid, self.pos)
        self.score = p
        for inf in info:
            (x_, y_), shape_, pos_ = inf
            index_ = shapePosto_index( shape_ , pos_ )
            if withinrange(x_,y_, NodeC.Shape[0], NodeC.Shape[1]):
                self.score += NodeC.Node_matrix[y_][x_].probability[index_]
            else:
                self.score = 0
                break

        return


def withinrange(x, y, row, col):
    if x >= 0 and x < col and y >= 0 and y < row:
        return True
    else:
        return False



def check_and_update(node, M, S, pid, Node_list):
    row, col = M.shape
    sign = True
    info, positions = one_to_threeotherpositions(node.x, node.y, node.shapeid, node.pos)
    for (x_, y_) in positions:
        if (not withinrange(x_,y_,row,col)) or M[y_][x_] != None:
            sign = False
            return sign

    #update and remove
    assert sign==True, 'Sign should be true to update'
    if node.shapeid==0:
        M[node.y][node.x] = (node.shapeid, 0)
    else:
        M[node.y][node.x] = (node.shapeid, pid)
    S[node.y][node.x] = (node.shapeid, node.pos)

    for inf in info:
        #update
        (x,y), shape, pos=inf
        M[y][x] = (shape, pid)
        S[y][x] = (shape, pos)
        #remove
        r_node = NodeC.Node_matrix[y][x]
        Node_list.remove( (-r_node.score, r_node.count, r_node) )

    return sign




















