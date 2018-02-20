from copy import deepcopy
from data_preprocessing import Goldenpositions,data_batch_iter_three,data_test_three
import numpy as np
from mysolution import Create_sample
import pickle
import multiprocessing
import tensorflow as tf
import time

GLOBAL_PARAMETERS={
    'gamesize': 125,
    'blank_range':[0.4],
    'simulation per move': 100,
    'width':20,
    'height':20,
    'batchsize':32,
    #'epoch per training': 1,
    'dataQ maxsize':20,
    'nframes': 1
}


def build_Policynetwork(height,width,nframes, n_resb_blocks, Tetris_filtering=False):
    def Tetris_weight(nframes):
        w = np.array([ [[0, 0, 0, 0],
                        [0, 1, 1, 0],
                        [0, 1, 1, 0],
                        [0, 0, 0, 0]],

                       [[1, 0, 0, 0],
                        [1, 0, 0, 0],
                        [1, 0, 0, 0],
                        [1, 0, 0, 0]],

                       [[1, 1, 1, 1],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0]],

                       [[1, 0, 0, 0],
                        [1, 0, 0, 0],
                        [1, 1, 0, 0],
                        [0, 0, 0, 0]],

                       [[0, 0, 1, 0],
                        [1, 1, 1, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0]],

                       [[1, 1, 0, 0],
                        [0, 1, 0, 0],
                        [0, 1, 0, 0],
                        [0, 0, 0, 0]],

                       [[1, 1, 1, 0],
                        [1, 0, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0]],

                       [[0, 1, 0, 0],
                        [0, 1, 0, 0],
                        [1, 1, 0, 0],
                        [0, 0, 0, 0]],

                       [[1, 1, 1, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0]],

                       [[1, 1, 0, 0],
                        [1, 0, 0, 0],
                        [1, 0, 0, 0],
                        [0, 0, 0, 0]],

                       [[1, 0, 0, 0],
                        [1, 1, 1, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0]],

                       [[1, 0, 0, 0],
                        [1, 1, 0, 0],
                        [1, 0, 0, 0],
                        [0, 0, 0, 0]],

                       [[0, 1, 0, 0],
                        [1, 1, 1, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0]],

                       [[0, 1, 0, 0],
                        [1, 1, 0, 0],
                        [0, 1, 0, 0],
                        [0, 0, 0, 0]],

                       [[1, 1, 1, 0],
                        [0, 1, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0]],

                       [[0, 1, 1, 0],
                        [1, 1, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0]],

                       [[1, 0, 0, 0],
                        [1, 1, 0, 0],
                        [0, 1, 0, 0],
                        [0, 0, 0, 0]],

                       [[1, 1, 0, 0],
                        [0, 1, 1, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0]],

                       [[0, 1, 0, 0],
                        [1, 1, 0, 0],
                        [1, 0, 0, 0],
                        [0, 0, 0, 0]],

                       ],dtype=np.float32)

        tetris_w = np.array([w for i in range(nframes)])
        tetris_w = tf.constant(tetris_w, dtype=tf.float32)
        tetris_w = tf.transpose(tetris_w, (2,3,0,1))
        return tetris_w

    myGraph = tf.Graph()
    with myGraph.as_default():
        x = tf.placeholder(tf.float32,shape= [None,nframes*height*width], name='input_puzzles')
        #y_v = tf.placeholder(tf.float32, shape = [None], name='values')
        y_p = tf.placeholder(tf.float32, shape=[None, 19], name='search_probabilities')

        is_training = tf.placeholder(tf.bool, name='is_training')
        learning_rate = tf.placeholder(tf.float32, shape=(), name='learning_rate')

        def conv_block(ipt, filter_kernalsize, channels_in, channels_out, is_training, name):
            with tf.name_scope(name):
                w = tf.Variable(tf.truncated_normal([filter_kernalsize[0], filter_kernalsize[1], channels_in,channels_out], stddev=0.1), name="w")
                conv = tf.nn.conv2d(ipt,w,strides=[1,1,1,1],padding="SAME",name="conv")
                bn = tf.layers.batch_normalization(
                    inputs=conv,
                    momentum=0.9,
                    training=is_training,
                    name=name+"_bn"
                )
                act = tf.nn.relu(bn,name='relu')
                return act

        def residual_block(ipt, channels, is_training, name):
            with tf.name_scope(name):
                w1 = tf.Variable(tf.truncated_normal([3,3,channels,channels], stddev=0.1), name="w1")
                conv1 = tf.nn.conv2d(ipt,w1, strides=[1,1,1,1], padding="SAME", name='conv1')
                bn1 = tf.layers.batch_normalization(
                    inputs=conv1,
                    momentum=0.9,
                    training=is_training,
                    name=name+"_bn1"
                )
                rl1 = tf.nn.relu(bn1, name="rl1")

                w2 =  tf.Variable(tf.truncated_normal([3,3,channels,channels], stddev=0.1), name="w2")
                conv2 = tf.nn.conv2d(rl1, w2, strides=[1,1,1,1], padding="SAME", name="conv2")
                bn2 = tf.layers.batch_normalization(
                    inputs=conv2,
                    momentum=0.9,
                    training=is_training,
                    name=name+"_bn2"
                )
                skip_conn = bn2 + ipt
                rl2 = tf.nn.relu(skip_conn, name='rl2')

                return rl2

        def linear_fullyconnect(ipt, in_dim, out_dim, name):
            with tf.name_scope(name):
                w = tf.Variable( tf.truncated_normal([in_dim, out_dim],stddev=0.1) , name="w")
                b = tf.Variable( tf.constant(0.1, shape=[out_dim]), name='bias' )
                h_linear = tf.matmul( ipt, w ) + b
                return h_linear



        x_puzzle = tf.transpose( tf.reshape(x,[-1, nframes, height, width]), (0,2,3,1) )

        if Tetris_filtering: # Filtering by tetris pieces
            tetris_w = Tetris_weight(nframes)
            x_puzzle = tf.nn.conv2d(x_puzzle,tetris_w, strides=[1,1,1,1], padding="SAME", name='tetris_conv') # tetris filtering by conv
            hidden = conv_block(x_puzzle, [3, 3], 19, 256, is_training, "conv_block")
        else:
            hidden = conv_block(x_puzzle, [3,3], nframes, 256, is_training,"conv_block")

        for i in range(1,n_resb_blocks+1):
            hidden = residual_block(hidden, 256, is_training, 'residual_block'+str(i) )

        #probability output
        hidden_prob = conv_block(hidden, [1,1], 256, 2, is_training, 'conv_block_prob' )
        flat_prob = tf.reshape(hidden_prob, [-1, width*height*2])
        logits = linear_fullyconnect(flat_prob, width*height*2, 19, 'fully_connect_prob')
        probabilities = tf.nn.softmax(logits, name='prob_predictions')
        cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=y_p, logits=logits), name='cross_entro'
        )
        correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y_p, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')

        #value output
        #hidden_value = conv_block(hidden, [1,1], 256, 1, is_training, 'conv_block_value')
        #fully connect to 256
        #hidden_flat = tf.reshape(hidden_value, [-1, width*height*1])
        #h_fc1 = tf.nn.relu( linear_fullyconnect(hidden_flat, width*height*1, 256, 'fully_connect_1') )
        #fully connect to scalar
        #h_fc2 = linear_fullyconnect(h_fc1, 256, 1, 'fully_connect_2')
        #y = tf.tanh(h_fc2)
        #y = tf.reshape(y, shape=[-1,], name='value_predictions')
        #mse = tf.reduce_mean(tf.squared_difference(y, y_v), name='MSError')

        lossL2 = tf.add_n([tf.nn.l2_loss(variable) for variable in tf.trainable_variables()
                           if 'bias' not in variable.name]) * 0.0001
        #loss = tf.add_n([mse, cross_entropy, lossL2], name='total_loss')
        loss = tf.add_n([cross_entropy, lossL2], name='total_loss')

        #summary
        tf.summary.scalar('loss',loss)
        merged = tf.summary.merge_all()

        #train
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss, name='train_policy')

    sess = tf.Session(graph=myGraph)
    assert sess.graph is myGraph
    return sess

def build_half_PV_network(height,width,nframes, n_resb_blocks, Tetris_filtering=False):
    def Tetris_weight(nframes):
        w = np.array([ [[0, 0, 0, 0],
                        [0, 1, 1, 0],
                        [0, 1, 1, 0],
                        [0, 0, 0, 0]],

                       [[1, 0, 0, 0],
                        [1, 0, 0, 0],
                        [1, 0, 0, 0],
                        [1, 0, 0, 0]],

                       [[1, 1, 1, 1],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0]],

                       [[1, 0, 0, 0],
                        [1, 0, 0, 0],
                        [1, 1, 0, 0],
                        [0, 0, 0, 0]],

                       [[0, 0, 1, 0],
                        [1, 1, 1, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0]],

                       [[1, 1, 0, 0],
                        [0, 1, 0, 0],
                        [0, 1, 0, 0],
                        [0, 0, 0, 0]],

                       [[1, 1, 1, 0],
                        [1, 0, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0]],

                       [[0, 1, 0, 0],
                        [0, 1, 0, 0],
                        [1, 1, 0, 0],
                        [0, 0, 0, 0]],

                       [[1, 1, 1, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0]],

                       [[1, 1, 0, 0],
                        [1, 0, 0, 0],
                        [1, 0, 0, 0],
                        [0, 0, 0, 0]],

                       [[1, 0, 0, 0],
                        [1, 1, 1, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0]],

                       [[1, 0, 0, 0],
                        [1, 1, 0, 0],
                        [1, 0, 0, 0],
                        [0, 0, 0, 0]],

                       [[0, 1, 0, 0],
                        [1, 1, 1, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0]],

                       [[0, 1, 0, 0],
                        [1, 1, 0, 0],
                        [0, 1, 0, 0],
                        [0, 0, 0, 0]],

                       [[1, 1, 1, 0],
                        [0, 1, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0]],

                       [[0, 1, 1, 0],
                        [1, 1, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0]],

                       [[1, 0, 0, 0],
                        [1, 1, 0, 0],
                        [0, 1, 0, 0],
                        [0, 0, 0, 0]],

                       [[1, 1, 0, 0],
                        [0, 1, 1, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0]],

                       [[0, 1, 0, 0],
                        [1, 1, 0, 0],
                        [1, 0, 0, 0],
                        [0, 0, 0, 0]],

                       ],dtype=np.float32)

        tetris_w = np.array([w for i in range(nframes)])
        tetris_w = tf.constant(tetris_w, dtype=tf.float32)
        tetris_w = tf.transpose(tetris_w, (2,3,0,1))
        return tetris_w

    myGraph = tf.Graph()
    with myGraph.as_default():
        x = tf.placeholder(tf.float32,shape= [None,nframes*height*width], name='input_puzzles')
        y_v = tf.placeholder(tf.float32, shape = [None], name='values')
        y_p = tf.placeholder(tf.float32, shape=[None, 19], name='search_probabilities')

        is_training = tf.placeholder(tf.bool, name='is_training')
        learning_rate = tf.placeholder(tf.float32, shape=(), name='learning_rate')

        def conv_block(ipt, filter_kernalsize, channels_in, channels_out, is_training, name):
            with tf.name_scope(name):
                w = tf.Variable(tf.truncated_normal([filter_kernalsize[0], filter_kernalsize[1], channels_in,channels_out], stddev=0.1), name="w")
                conv = tf.nn.conv2d(ipt,w,strides=[1,1,1,1],padding="SAME",name="conv")
                bn = tf.layers.batch_normalization(
                    inputs=conv,
                    momentum=0.9,
                    training=is_training,
                    name=name+"_bn"
                )
                act = tf.nn.relu(bn,name='relu')
                return act

        def residual_block(ipt, channels, is_training, name):
            with tf.name_scope(name):
                w1 = tf.Variable(tf.truncated_normal([3,3,channels,channels], stddev=0.1), name="w1")
                conv1 = tf.nn.conv2d(ipt,w1, strides=[1,1,1,1], padding="SAME", name='conv1')
                bn1 = tf.layers.batch_normalization(
                    inputs=conv1,
                    momentum=0.9,
                    training=is_training,
                    name=name+"_bn1"
                )
                rl1 = tf.nn.relu(bn1, name="rl1")

                w2 =  tf.Variable(tf.truncated_normal([3,3,channels,channels], stddev=0.1), name="w2")
                conv2 = tf.nn.conv2d(rl1, w2, strides=[1,1,1,1], padding="SAME", name="conv2")
                bn2 = tf.layers.batch_normalization(
                    inputs=conv2,
                    momentum=0.9,
                    training=is_training,
                    name=name+"_bn2"
                )
                skip_conn = bn2 + ipt
                rl2 = tf.nn.relu(skip_conn, name='rl2')

                return rl2

        def linear_fullyconnect(ipt, in_dim, out_dim, name):
            with tf.name_scope(name):
                w = tf.Variable( tf.truncated_normal([in_dim, out_dim],stddev=0.1) , name="w")
                b = tf.Variable( tf.constant(0.1, shape=[out_dim]), name='bias' )
                h_linear = tf.matmul( ipt, w ) + b
                return h_linear



        x_puzzle = tf.transpose( tf.reshape(x,[-1, nframes, height, width]), (0,2,3,1) )

        if Tetris_filtering: # Filtering by tetris pieces
            tetris_w = Tetris_weight(nframes)
            x_puzzle = tf.nn.conv2d(x_puzzle,tetris_w, strides=[1,1,1,1], padding="SAME", name='tetris_conv') # tetris filtering by conv
            hidden = conv_block(x_puzzle, [3, 3], 19, 256, is_training, "conv_block")
        else:
            hidden = conv_block(x_puzzle, [3,3], nframes, 256, is_training,"conv_block")

        for i in range(1,n_resb_blocks+1):
            hidden = residual_block(hidden, 256, is_training, 'residual_block'+str(i) )

        #probability output
        hidden_prob = conv_block(hidden, [1,1], 256, 2, is_training, 'conv_block_prob' )
        flat_prob = tf.reshape(hidden_prob, [-1, width*height*2])
        logits = linear_fullyconnect(flat_prob, width*height*2, 19, 'fully_connect_prob')
        probabilities = tf.nn.softmax(logits, name='prob_predictions')
        cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=y_p, logits=logits), name='cross_entro'
        )

        #value output
        hidden_value = conv_block(hidden, [1,1], 256, 1, is_training, 'conv_block_value')
        #fully connect to 256
        hidden_flat = tf.reshape(hidden_value, [-1, width*height*1])
        h_fc1 = tf.nn.relu( linear_fullyconnect(hidden_flat, width*height*1, 256, 'fully_connect_1') )
        #fully connect to scalar
        h_fc2 = linear_fullyconnect(h_fc1, 256, 1, 'fully_connect_2')
        y = tf.tanh(h_fc2)
        y = tf.reshape(y, shape=[-1,], name='value_predictions')
        mse = tf.reduce_mean(tf.squared_difference(y, y_v), name='MSError')

        lossL2 = tf.add_n([tf.nn.l2_loss(variable) for variable in tf.trainable_variables()
                           if 'bias' not in variable.name]) * 0.0001
        loss = tf.add_n([0.5*mse, cross_entropy, lossL2], name='total_loss')

        #summary
        tf.summary.scalar('loss',loss)
        merged = tf.summary.merge_all()

        #train
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss, name='train_mini')

    sess = tf.Session(graph=myGraph)
    assert sess.graph is myGraph
    return sess

def build_Valuenetwork(height,width,nframes, n_resb_blocks, Tetris_filtering=False):
    def Tetris_weight(nframes):
        w = np.array([ [[0, 0, 0, 0],
                        [0, 1, 1, 0],
                        [0, 1, 1, 0],
                        [0, 0, 0, 0]],

                       [[1, 0, 0, 0],
                        [1, 0, 0, 0],
                        [1, 0, 0, 0],
                        [1, 0, 0, 0]],

                       [[1, 1, 1, 1],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0]],

                       [[1, 0, 0, 0],
                        [1, 0, 0, 0],
                        [1, 1, 0, 0],
                        [0, 0, 0, 0]],

                       [[0, 0, 1, 0],
                        [1, 1, 1, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0]],

                       [[1, 1, 0, 0],
                        [0, 1, 0, 0],
                        [0, 1, 0, 0],
                        [0, 0, 0, 0]],

                       [[1, 1, 1, 0],
                        [1, 0, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0]],

                       [[0, 1, 0, 0],
                        [0, 1, 0, 0],
                        [1, 1, 0, 0],
                        [0, 0, 0, 0]],

                       [[1, 1, 1, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0]],

                       [[1, 1, 0, 0],
                        [1, 0, 0, 0],
                        [1, 0, 0, 0],
                        [0, 0, 0, 0]],

                       [[1, 0, 0, 0],
                        [1, 1, 1, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0]],

                       [[1, 0, 0, 0],
                        [1, 1, 0, 0],
                        [1, 0, 0, 0],
                        [0, 0, 0, 0]],

                       [[0, 1, 0, 0],
                        [1, 1, 1, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0]],

                       [[0, 1, 0, 0],
                        [1, 1, 0, 0],
                        [0, 1, 0, 0],
                        [0, 0, 0, 0]],

                       [[1, 1, 1, 0],
                        [0, 1, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0]],

                       [[0, 1, 1, 0],
                        [1, 1, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0]],

                       [[1, 0, 0, 0],
                        [1, 1, 0, 0],
                        [0, 1, 0, 0],
                        [0, 0, 0, 0]],

                       [[1, 1, 0, 0],
                        [0, 1, 1, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0]],

                       [[0, 1, 0, 0],
                        [1, 1, 0, 0],
                        [1, 0, 0, 0],
                        [0, 0, 0, 0]],

                       ],dtype=np.float32)

        tetris_w = np.array([w for i in range(nframes)])
        tetris_w = tf.constant(tetris_w, dtype=tf.float32)
        tetris_w = tf.transpose(tetris_w, (2,3,0,1))
        return tetris_w

    myGraph = tf.Graph()
    with myGraph.as_default():
        x = tf.placeholder(tf.float32,shape= [None,nframes*height*width], name='input_puzzles')
        y_v = tf.placeholder(tf.float32, shape = [None], name='values')
        #y_p = tf.placeholder(tf.float32, shape=[None, 19], name='search_probabilities')

        is_training = tf.placeholder(tf.bool, name='is_training')
        learning_rate = tf.placeholder(tf.float32, shape=(), name='learning_rate')

        def conv_block(ipt, filter_kernalsize, channels_in, channels_out, is_training, name):
            with tf.name_scope(name):
                w = tf.Variable(tf.truncated_normal([filter_kernalsize[0], filter_kernalsize[1], channels_in,channels_out], stddev=0.1), name="w")
                conv = tf.nn.conv2d(ipt,w,strides=[1,1,1,1],padding="SAME",name="conv")
                bn = tf.layers.batch_normalization(
                    inputs=conv,
                    momentum=0.9,
                    training=is_training,
                    name=name+"_bn"
                )
                act = tf.nn.relu(bn,name='relu')
                return act

        def residual_block(ipt, channels, is_training, name):
            with tf.name_scope(name):
                w1 = tf.Variable(tf.truncated_normal([3,3,channels,channels], stddev=0.1), name="w1")
                conv1 = tf.nn.conv2d(ipt,w1, strides=[1,1,1,1], padding="SAME", name='conv1')
                bn1 = tf.layers.batch_normalization(
                    inputs=conv1,
                    momentum=0.9,
                    training=is_training,
                    name=name+"_bn1"
                )
                rl1 = tf.nn.relu(bn1, name="rl1")

                w2 =  tf.Variable(tf.truncated_normal([3,3,channels,channels], stddev=0.1), name="w2")
                conv2 = tf.nn.conv2d(rl1, w2, strides=[1,1,1,1], padding="SAME", name="conv2")
                bn2 = tf.layers.batch_normalization(
                    inputs=conv2,
                    momentum=0.9,
                    training=is_training,
                    name=name+"_bn2"
                )
                skip_conn = bn2 + ipt
                rl2 = tf.nn.relu(skip_conn, name='rl2')

                return rl2

        def linear_fullyconnect(ipt, in_dim, out_dim, name):
            with tf.name_scope(name):
                w = tf.Variable( tf.truncated_normal([in_dim, out_dim],stddev=0.1) , name="w")
                b = tf.Variable( tf.constant(0.1, shape=[out_dim]), name='bias' )
                h_linear = tf.matmul( ipt, w ) + b
                return h_linear



        x_puzzle = tf.transpose( tf.reshape(x,[-1, nframes, height, width]), (0,2,3,1) )

        if Tetris_filtering: # Filtering by tetris pieces
            tetris_w = Tetris_weight(nframes)
            x_puzzle = tf.nn.conv2d(x_puzzle,tetris_w, strides=[1,1,1,1], padding="SAME", name='tetris_conv') # tetris filtering by conv
            hidden = conv_block(x_puzzle, [3, 3], 19, 256, is_training, "conv_block")
        else:
            hidden = conv_block(x_puzzle, [3,3], nframes, 256, is_training,"conv_block")

        for i in range(1,n_resb_blocks+1):
            hidden = residual_block(hidden, 256, is_training, 'residual_block'+str(i) )

        #probability output
        #hidden_prob = conv_block(hidden, [1,1], 256, 2, is_training, 'conv_block_prob' )
        #flat_prob = tf.reshape(hidden_prob, [-1, width*height*2])
        #logits = linear_fullyconnect(flat_prob, width*height*2, 19, 'fully_connect_prob')
        #probabilities = tf.nn.softmax(logits, name='prob_predictions')
        #cross_entropy = tf.reduce_mean(
        #    tf.nn.softmax_cross_entropy_with_logits(labels=y_p, logits=logits), name='cross_entro'
        #)

        #value output
        hidden_value = conv_block(hidden, [1,1], 256, 1, is_training, 'conv_block_value')
        #fully connect to 256
        hidden_flat = tf.reshape(hidden_value, [-1, width*height*1])
        h_fc1 = tf.nn.relu( linear_fullyconnect(hidden_flat, width*height*1, 256, 'fully_connect_1') )
        #fully connect to scalar
        h_fc2 = linear_fullyconnect(h_fc1, 256, 1, 'fully_connect_2')
        y = tf.tanh(h_fc2)
        y = tf.reshape(y, shape=[-1,], name='value_predictions')
        mse = tf.reduce_mean(tf.squared_difference(y, y_v), name='MSError')

        lossL2 = tf.add_n([tf.nn.l2_loss(variable) for variable in tf.trainable_variables()
                           if 'bias' not in variable.name]) * 0.0001
        #loss = tf.add_n([mse, cross_entropy, lossL2], name='total_loss')
        loss = tf.add_n([mse, lossL2], name='total_loss')

        #summary
        tf.summary.scalar('loss',loss)
        merged = tf.summary.merge_all()

        #train
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss, name='train_value')

    sess = tf.Session(graph=myGraph)
    assert sess.graph is myGraph
    return sess


class Edge(object):
    def __init__(self,start,end, P, Prob_id):
        self.start = start
        self.end = end
        self.N = 0
        self.Q = 0.0
        self.W = 0.0
        self.P = P
        self.Prob_id = Prob_id


class Node(object):

    def __init__(self,state,table,cumR,uniR, sess):
        self.table = table
        self.state = state
        self.cumR = cumR
        self.uniR = uniR
        self.sess = sess
        self.expanded = False
        self.row = self.state.shape[0]
        self.col = self.state.shape[1]

    def check_explore(self):
        """
        self.terminal indicates whether this state is terminal
        self.V is real value: -1 for unsuccessful terminal and 1 for successful terminal
        self.score facilitate the MCTS by considering the depth of the terminal
        """
        self.expanded = True

        for y in range(0,self.row):
            for x in range(0,self.col):
                if self.state[y][x]==1:
                    self.x = x
                    self.y = y
                    self.fetch_edges(x,y)
                    if len(self.edges)>0:
                        self.terminal = False
                        #self.V = Vs[0]
                        return
                    else:
                        self.terminal = True
                        self.score = -1.0 + self.cumR
                        self.V = -1.0
                        return

        self.x = self.col - 1
        self.y = self.row - 1
        self.terminal = True
        self.score = 1.0
        self.V = 1.0
        return

    def fetch_edges(self,x,y):
        self.edges = []
        for shape in range(1,20):
            available = True
            child_state = deepcopy(self.state)
            for p in [0, 1, 2, 3]:
                (x_, y_) = Goldenpositions[shape][p] - Goldenpositions[shape][0] + np.array([x, y])
                if withinrange(x_,y_,self.row,self.col) and child_state[y_][x_] == 1:
                    child_state[y_][x_] = 0
                    continue
                else:
                    available = False
                    break

            if available:
                stateid = child_state.tostring()
                if stateid in self.table:
                    #append edge
                    ed = Edge(self, self.table[stateid], 1.0, shape-1 )
                    self.edges.append(ed)
                else:
                    # create node
                    c_node = Node(child_state,self.table, self.cumR+self.uniR, self.uniR, self.sess)
                    # add in table
                    self.table[stateid] = c_node
                    #append edge
                    ed = Edge(self, c_node, 1.0, shape-1)
                    self.edges.append(ed)

        return


def withinrange(x, y, row, col):
    if x >= 0 and x < col and y >= 0 and y < row:
        return True
    else:
        return False

class Simulation(object):
    """
    We generate training samples by using real move state rather than simulation move state,
    because one simulation state may have multiple results to be either 1 or -1.
     This would not be efficient in supervised learning
    """

    def __init__(self,rootnode, sess , nframe):
        self.nframe = nframe
        self.currentnode = rootnode
        self.sess = sess
        self.path = []



    def run(self):

        while True:
            if not self.currentnode.expanded:
                self.currentnode.check_explore()
            if self.currentnode.terminal:
                self.backup(self.currentnode.score)
                return
            else:
                edge = self.selectfrom(self.currentnode)
                self.path.append(edge)
                self.currentnode = edge.end



    def backup(self,v):
        for edge in self.path:
            edge.W += v
            edge.N += 1
            edge.Q = edge.W/edge.N
        return


    def selectfrom(self,node):
        sum_N = np.sum([ edge.N for edge in node.edges ]) + 1
        value_max = (float('-inf'), None)
        for edge in node.edges:
            v = edge.Q + ( ( edge.P * np.sqrt(sum_N) ) / (1 + edge.N) )
            if v > value_max[0]:
                value_max = (v, edge)

        return value_max[1]


class Game(object):

    def __init__(self, target, n_search, nframes, sess):
        self.nframes = nframes
        self.target = np.array(target)
        self.table = {}
        self.n_search = n_search
        self.sess = sess

        self.current_realnode = Node(deepcopy(self.target),self.table, 0.0, 4/np.sum(self.target), self.sess)
        self.table[self.current_realnode.state.tostring()] = self.current_realnode
        self.real_nodepath = [self.current_realnode]
        self.search_Ps = []

    def play(self):

        while True:
            self.play_one_move(self.current_realnode)

            if not self.current_realnode.expanded:
                self.current_realnode.check_explore()

            if self.current_realnode.terminal:
                search_prob = np.zeros(19, dtype=np.float32)
                self.search_Ps.append(search_prob)
                assert len(self.search_Ps)==len(self.real_nodepath), 'size of search prob not equal to real_nodepath'
                gamedata = []
                shape_dim = self.current_realnode.state.shape
                for i, node in enumerate(self.real_nodepath):
                    loc_frame = np.zeros(shape_dim, dtype=np.int)
                    loc_frame[node.y][node.x] = 1
                    if i == 0:
                        gamedata.append((np.reshape([node.state,loc_frame], [-1]), 1.0, self.search_Ps[i]))
                    else:
                        gamedata.append((np.reshape([node.state,loc_frame], [-1]), self.current_realnode.V, self.search_Ps[i]))

                return gamedata, self.current_realnode.V, self.current_realnode.score



    def play_one_move(self,startnode):
        for i in range(0,self.n_search):
            simulation = Simulation(startnode, self.sess, self.nframes)
            simulation.run()


        (maxQ,maxedge) = max([(edge.Q, edge) for edge in startnode.edges], key=lambda s:s[0])
        search_prob = np.zeros(19, dtype=np.float32)
        #search_prob[maxedge.Prob_id] = 1.0
        self.search_Ps.append( search_prob )

        self.current_realnode = maxedge.end
        self.real_nodepath.append(self.current_realnode)

        return


def play_to_the_end(target, first_round, rightdata, info, nframes, eval_sess,
                    n_search=GLOBAL_PARAMETERS['simulation per move']):

    game = Game(target, n_search, nframes, eval_sess)
    gamedata, result, score = game.play()

    if first_round:
        info['first_score'] = score

    if result > 0:
        return
    else:
        for i in range(len(gamedata) - 1, -1, -1):
            if not np.array_equal(gamedata[i][0], rightdata[i][0]):
                info['Data'].append(gamedata[i])
                info['Data'].append(rightdata[i])
            else:
                new_pos = len(gamedata) - 1
                break
        newtarget = np.reshape(rightdata[new_pos][0], [2, 20, 20]).astype(np.int)[0]
        play_to_the_end(newtarget, False, rightdata[new_pos:], info, nframes, eval_sess)
        return

def play_games(eval_sess, nframes, name,
               prob_blank_range=GLOBAL_PARAMETERS['blank_range'],
               height=GLOBAL_PARAMETERS['height'],
               width=GLOBAL_PARAMETERS['width'],
               ):

    Data = []
    n_game = 0
    scores = []
    print('Play games...:')
    while len(Data) < 100000:
        for prob_blank in prob_blank_range:
            sample = Create_sample(height, width, prob_blank)
            sample.add_pieces()
            target, solution = sample.T, sample.S
            rightdata = solve_game(target, solution)
            info = {'Data': []}
            play_to_the_end(target, True, rightdata, info, nframes, eval_sess)
            Data.extend(info['Data'])
            n_game += 1
            scores.append(info['first_score'])
            print('game {}th, data {}, score {}'.format(n_game, len(info['Data']), info['first_score']))

    with open('supervised_PV_data_{}.pickle'.format(name), 'wb') as dfile:
        pickle.dump(Data, dfile)
    scores=np.array(scores)
    return {'mean score': scores.mean(), 'std':scores.std()}


def PolicyValue_data(start,end):
    pool = multiprocessing.Pool(end-start)
    results = pool.starmap(play_games, [(None, None, i,) for i in range(start,end)])
    for result in results:
        print(result)
    print('Finish Done')
    return

def train_PolicyValue(trainstart,trainend,testid, PVmodel, min_mse, min_cross):
    print('traning from dataset {} to dataset {}'.format(trainstart, trainend-1))
    print('test dataset {}'.format(testid))
    train_names = []
    for i in range(trainstart,trainend):
        train_names.append(i)

    with open('supervised_PV_data_{}.pickle'.format(testid), 'rb') as dfile:
        test_data = pickle.load(dfile)
    test_data = np.array(test_data)
    shuffle_indices = np.random.permutation(np.arange(len(test_data)))
    shuffled_data = test_data[shuffle_indices]
    test_data =shuffled_data[:2000]
    test_x, test_y, test_p = data_test_three(test_data)
    sess = build_half_PV_network(20, 20, 2, 10, False)
    min_mse, min_cross = min_mse, min_cross

    with sess:
        saver = tf.train.Saver()
        saver.restore(sess, './{}.ckpt'.format(PVmodel))
        time_step = 0
        epo_num = 1

        while True:

            np.random.shuffle(train_names)
            print('start epoch {}, shuffle training dataset: {}'.format(epo_num, train_names))
            epo_step = 0
            for name in train_names:
                print('training dataset {} ......'.format(name))
                with open('supervised_PV_data_{}.pickle'.format(name), 'rb') as dfile:
                    training_data = pickle.load(dfile)
                batches = data_batch_iter_three(training_data, 64, 1)
                for inputdata, outputdata, probdata in batches:
                    if epo_step % 100 == 0:
                        train_mse, train_crossentro = sess.run(['MSError:0', 'cross_entro:0'], feed_dict={
                            'input_puzzles:0': inputdata.astype(np.float32),
                            'values:0': outputdata,
                            'search_probabilities:0': probdata,
                            'is_training:0': False
                        })
                        print(time.strftime("%Y-%m-%d %H:%M:%S"),
                              ': total training step {}, epo step {}, train mse {}, train crossentro {}'.format(
                                  time_step, epo_step,  train_mse, train_crossentro))

                    if epo_step % 4000 == 0:
                        test_mse, test_crossentro = sess.run(['MSError:0', 'cross_entro:0'], feed_dict={
                            'input_puzzles:0': test_x.astype(np.float32),
                            'values:0': test_y,
                            'search_probabilities:0': test_p,
                            'is_training:0': False
                        })
                        print(time.strftime("%Y-%m-%d %H:%M:%S"),
                              ': total step %d, epo step %d, test loss %g (test mse %g, test cross entro %g)' % ( time_step,
                              epo_step,  test_mse+test_crossentro, test_mse, test_crossentro))
                        if test_mse < min_mse and test_crossentro < min_cross:
                            min_mse = test_mse
                            min_cross = test_crossentro
                            save_tain = saver.save(sess, './{}.ckpt'.format(PVmodel))
                            print(time.strftime("%Y-%m-%d %H:%M:%S"),
                              ' : train_model saved in file: {} '.format(save_tain))

                    sess.run('train_mini', feed_dict={
                        'input_puzzles:0': inputdata.astype(np.float32),
                        'values:0': outputdata,
                        'search_probabilities:0': probdata,
                        'is_training:0': True,
                        'learning_rate:0': 1e-4
                    })
                    time_step += 1
                    epo_step += 1
            epo_num += 1


def train_value(trainstart,trainend,testid, valuemodel, minloss):
    print('traning from dataset {} to dataset {}'.format(trainstart, trainend-1))
    print('test dataset {}'.format(testid))
    train_names = []
    for i in range(trainstart,trainend):
        train_names.append(i)

    with open('supervised_PV_data_{}.pickle'.format(testid), 'rb') as dfile:
        test_data = pickle.load(dfile)
    test_data = np.array(test_data)
    shuffle_indices = np.random.permutation(np.arange(len(test_data)))
    shuffled_data = test_data[shuffle_indices]
    test_data =shuffled_data[:2000]
    test_x, test_y, test_p = data_test_three(test_data)
    sess = build_Valuenetwork(20, 20, 2, 10, False)
    minloss = minloss

    with sess:
        saver = tf.train.Saver()
        saver.restore(sess, './{}.ckpt'.format(valuemodel))
        time_step = 0
        epo_num = 1

        while True:

            np.random.shuffle(train_names)
            print('start epoch {}, shuffle training dataset: {}'.format(epo_num, train_names))
            epo_step = 0
            for name in train_names:
                print('training dataset {} ......'.format(name))
                with open('supervised_PV_data_{}.pickle'.format(name), 'rb') as dfile:
                    training_data = pickle.load(dfile)
                batches = data_batch_iter_three(training_data, 64, 1)
                for inputdata, outputdata, probdata in batches:
                    if epo_step % 100 == 0:
                        train_mse = sess.run('MSError:0', feed_dict={
                            'input_puzzles:0': inputdata.astype(np.float32),
                            'values:0': outputdata,
                            'is_training:0': False
                        })
                        print(time.strftime("%Y-%m-%d %H:%M:%S"),
                              ': total training step {}, epo step {}, train mse {}'.format(
                                  time_step, epo_step,  train_mse))

                    if epo_step % 4000 == 0:
                        test_mse = sess.run('MSError:0', feed_dict={
                            'input_puzzles:0': test_x.astype(np.float32),
                            'values:0': test_y,
                            'is_training:0': False
                        })
                        print(time.strftime("%Y-%m-%d %H:%M:%S"),
                              ': total step %d, epo step %d, test mse %g' % ( time_step,
                              epo_step,  test_mse))
                        if test_mse < minloss:
                            minloss = test_mse
                            save_tain = saver.save(sess, './{}.ckpt'.format(valuemodel))
                            print(time.strftime("%Y-%m-%d %H:%M:%S"),
                              ' : train_model saved in file: {} '.format(save_tain))

                    sess.run('train_value', feed_dict={
                        'input_puzzles:0': inputdata.astype(np.float32),
                        'values:0': outputdata,
                        'is_training:0': True,
                        'learning_rate:0': 1e-4
                    })
                    time_step += 1
                    epo_step += 1
            epo_num += 1

def solve_game(T, S):
    """
    Generate successful game data solving the target
    :param T: target
    :param S: solution of the target
    :return:
    """
    target = deepcopy(T)
    solution = deepcopy(S)
    row = T.shape[0]
    col = T.shape[1]

    data = []


    for y in range(0, row):
        for x in range(0, col):
            if target[y][x] == 1:
                (shape,pos) = solution[y][x]
                assert pos==0, 'top-left corner pos should be 0'
                search_prob = np.zeros(19, dtype=np.float32)
                search_prob[shape - 1] = 1.0
                loc_frame = np.zeros([row,col],dtype=np.int)
                loc_frame[y][x] = 1
                data.append((np.reshape(deepcopy([target,loc_frame]), [-1]), 1.0, search_prob))

                # next node
                for p in [0, 1, 2, 3]:
                    (x_, y_) = Goldenpositions[shape][p] - Goldenpositions[shape][0] + np.array([x, y])
                    assert withinrange(x_,y_,row,col),'x,y not withinrange'
                    assert target[y_][x_]==1, '(x,y) at target not to be 1'
                    target[y_][x_] = 0

    assert np.sum(target)==0, 'final right data not be blank'
    search_prob = np.zeros(19, dtype=np.float32)
    loc_frame = np.zeros([row,col],dtype=np.int)
    loc_frame[row-1][col-1] = 1
    data.append((np.reshape(deepcopy([target,loc_frame]), [-1]), 1.0, search_prob))

    return data


def policy_data_process(size, name):
    height = GLOBAL_PARAMETERS['height']
    width = GLOBAL_PARAMETERS['width']
    Data = []
    n_game = 0
    while len(Data) < size:
        for prob_blank in GLOBAL_PARAMETERS['blank_range']:
            sample = Create_sample(height, width, prob_blank)
            sample.add_pieces()
            target, solution = sample.T, sample.S
            rightdata = solve_game(target, solution)
            cleandata = rightdata[:-1]
            Data.extend(cleandata)
            n_game += 1
            print('game {}th, data {}'.format(n_game, len(cleandata)))

    with open('supervised_policy_data_{}.pickle'.format(name), 'wb') as dfile:
        pickle.dump(Data, dfile)
    print('Finish {}'.format(name))
    return

def Policy_data(start,end):
    pool = multiprocessing.Pool(end-start)
    pool.starmap(policy_data_process, [(100000,i) for i in range(start,end)])
    print('Finish Done')
    return


def train_policy(trainstart,trainend,testid, policymodel, minloss):
    print('traning from dataset {} to dataset {}'.format(trainstart, trainend-1))
    print('test dataset {}'.format(testid))
    train_names = []
    for i in range(trainstart,trainend):
        train_names.append(i)

    with open('supervised_policy_data_{}.pickle'.format(testid), 'rb') as dfile:
        test_data = pickle.load(dfile)
    test_data = np.array(test_data)
    shuffle_indices = np.random.permutation(np.arange(len(test_data)))
    shuffled_data = test_data[shuffle_indices]
    test_data =shuffled_data[:2000]
    test_x, test_y, test_p = data_test_three(test_data)
    sess = build_Policynetwork(20,20,2,10,False)
    minloss = minloss

    with sess:
        saver = tf.train.Saver()
        saver.restore(sess, './{}.ckpt'.format(policymodel))
        time_step = 0
        epo_num = 1

        while True:

            np.random.shuffle(train_names)
            print('start epoch {}, shuffle training dataset: {}'.format(epo_num, train_names))
            epo_step = 0
            for name in train_names:
                print('training dataset {} ......'.format(name))
                with open('supervised_policy_data_{}.pickle'.format(name), 'rb') as dfile:
                    training_data = pickle.load(dfile)
                batches = data_batch_iter_three(training_data, 64, 1)
                for inputdata, outputdata, probdata in batches:
                    if epo_step % 100 == 0:
                        train_acc, train_crossentro = sess.run(['accuracy:0', 'cross_entro:0'], feed_dict={
                            'input_puzzles:0': inputdata.astype(np.float32),
                            'search_probabilities:0': probdata,
                            'is_training:0': False
                        })
                        print(time.strftime("%Y-%m-%d %H:%M:%S"),
                              ': total training step {}, epo step {}, train acc {}, train crossentro {}'.format(
                                  time_step, epo_step,  train_acc, train_crossentro))

                    if epo_step % 4000 == 0:
                        test_acc, test_crossentro = sess.run(['accuracy:0', 'cross_entro:0'], feed_dict={
                            'input_puzzles:0': test_x.astype(np.float32),
                            'search_probabilities:0': test_p,
                            'is_training:0': False
                        })
                        print(time.strftime("%Y-%m-%d %H:%M:%S"),
                              ': total step %d, epo step %d, test acc %g, test cross entro %g' % ( time_step,
                              epo_step,  test_acc, test_crossentro))
                        if test_crossentro < minloss:
                            minloss = test_crossentro
                            save_tain = saver.save(sess, './{}.ckpt'.format(policymodel))
                            print(time.strftime("%Y-%m-%d %H:%M:%S"),
                              ' : train_model saved in file: {} '.format(save_tain))

                    sess.run('train_policy', feed_dict={
                        'input_puzzles:0': inputdata.astype(np.float32),
                        'search_probabilities:0': probdata,
                        'is_training:0': True,
                        'learning_rate:0': 1e-4
                    })
                    time_step += 1
                    epo_step += 1
            epo_num += 1





