# Same algorithm
# 1. Use history
# 2. Use equal size data
# 3. Backup terminal value, and predict_value evevy L step

import numpy as np
import tensorflow as tf
from data_preprocessing import Goldenpositions,data_batch_iter
from copy import deepcopy
from mysolution import Create_sample
import time
import pickle
import math

GLOBAL_PARAMETERS={
    'N games per density':10,
    'blank_range':[0.4],
    'simulation per move': 100,
    'width':20,
    'height':20,
    'batchsize':32,
    #'epoch per training': 1,
    'dataQ maxsize':20
}


def old_build_neuralnetwork(height,width,nframes):
    myGraph = tf.Graph()
    with myGraph.as_default():
        x = tf.placeholder(tf.float32,shape= [None,nframes*height*width], name='input_puzzles')
        y_ = tf.placeholder(tf.float32, shape = [None], name='labels')
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
        hidden = conv_block(x_puzzle, [3,3], nframes, 256, is_training,"conv_block")

        for i in range(1,5):
            hidden = residual_block(hidden, 256, is_training, 'residual_block'+str(i) )

        #output conv
        hidden = conv_block(hidden, [1,1], 256, 1, is_training, 'conv_block_out')

        #fully connect to 256
        hidden_flat = tf.reshape(hidden, [-1, width*height*1])
        h_fc1 = tf.nn.relu( linear_fullyconnect(hidden_flat, width*height*1, 256, 'fully_connect_1') )

        #fully connect to scalar
        h_fc2 = linear_fullyconnect(h_fc1, 256, 1, 'fully_connect_2')
        y = tf.tanh(h_fc2)
        y = tf.reshape(y, shape=[-1,], name='predictions')

        lossL2 = tf.add_n([tf.nn.l2_loss(variable) for variable in tf.trainable_variables()
                           if 'bias' not in variable.name]) * 0.0001

        mse = tf.reduce_mean( tf.squared_difference(y,y_), name='MSError' )
        loss = tf.add(mse, lossL2, name='total_loss')

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

def build_neuralnetwork(height,width,nframes, n_resb_blocks, Tetris_filtering=False):
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
        y_ = tf.placeholder(tf.float32, shape = [None], name='labels')
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

        #output conv
        hidden = conv_block(hidden, [1,1], 256, 1, is_training, 'conv_block_out')

        #fully connect to 256
        hidden_flat = tf.reshape(hidden, [-1, width*height*1])
        h_fc1 = tf.nn.relu( linear_fullyconnect(hidden_flat, width*height*1, 256, 'fully_connect_1') )

        #fully connect to scalar
        h_fc2 = linear_fullyconnect(h_fc1, 256, 1, 'fully_connect_2')
        y = tf.tanh(h_fc2)
        y = tf.reshape(y, shape=[-1,], name='predictions')

        lossL2 = tf.add_n([tf.nn.l2_loss(variable) for variable in tf.trainable_variables()
                           if 'bias' not in variable.name]) * 0.0001

        mse = tf.reduce_mean( tf.squared_difference(y,y_), name='MSError' )
        loss = tf.add(mse, lossL2, name='total_loss')

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

        self.N = 0
        self.Q = 0.0
        self.W = 0.0



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
                    self.fetch_children(x,y)
                    if len(self.children)>0:
                        self.terminal = False
                        self.V = self.sess.run('predictions:0',
                                                      feed_dict={'input_puzzles:0': np.reshape(self.state, [ 1, self.row*self.col] ).astype(np.float32),
                                                                 'is_training:0': False}
                                              )[0]
                        return
                    else:
                        self.terminal = True
                        self.score = -1.0 + self.cumR
                        self.V = -1.0
                        return

        self.terminal = True
        self.score = 1.0
        self.V = 1.0
        return

    def fetch_children(self,x,y):
        self.children = []
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
                    #append child
                    self.children.append( self.table[stateid] )
                else:
                    # create child
                    c_node = Node(child_state,self.table, self.cumR+self.uniR, self.uniR, self.sess)
                    # add in table
                    self.table[stateid] = c_node
                    #append child
                    self.children.append(c_node)

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
        self.path = [rootnode]



    def run(self):

        while True:
            if not self.currentnode.expanded:
                self.currentnode.check_explore()
                self.backup(self.currentnode.V)
                return
            else:
                if self.currentnode.terminal:
                    self.backup(self.currentnode.V)
                    return
                else:
                    self.currentnode = self.selectfrom(self.currentnode)
                    self.path.append(self.currentnode)


            #if self.t > 0 and (self.t%self.L==0):
            #    if self.nframe > len(self.path):
            #        history = [ np.zeros([ self.currentnode.row, self.currentnode.col]) for i in range(self.nframe - len(self.path)) ]
            #        for node in self.path:
            #            history.append( node.state )
            #    else:
            #        history = [ node.state for node in self.path[-self.nframe:] ]

            #    predict_value = self.sess.run('predictions:0',
            #                                  feed_dict={'input_puzzles:0':np.reshape(history, [1,self.nframe*self.currentnode.row*self.currentnode.col]).astype(np.float32),
            #                                             'is_training:0': False}
            #                                  )
            #    self.backup(predict_value[0])
                #self.backup(0.0)
                #for child in self.currentnode.children:
                #    child.N = 0
                #    child.Q = 0.0
                #    child.W = 0.0



    def backup(self,v):
        for node in self.path:
            node.W += v
            node.N += 1
            node.Q = node.W/node.N
        return


    def selectfrom(self,node):
        sum_N = np.sum([ child.N for child in node.children ])
        value_max = (float('-inf'), None)
        for child in node.children:
            v = child.Q + ( ( np.sqrt(2*sum_N) ) / (1 + child.N) )
            if v > value_max[0]:
                value_max = (v, child)

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

    def play(self):

        while True:
            self.play_one_move(self.current_realnode)

            if self.current_realnode.terminal:
                gamedata = []
                for i, node in enumerate(self.real_nodepath):
                    if i==0:
                        gamedata.append( (np.reshape(node.state, [-1]), 1.0) )
                    else:
                        gamedata.append( (np.reshape(node.state, [-1]), self.current_realnode.V) )

                return gamedata, self.current_realnode.V, self.current_realnode.score



    def play_one_move(self,startnode):
        for i in range(0,self.n_search):
            simulation = Simulation(startnode, self.sess, self.nframes)
            simulation.run()


        (maxQ,maxchild) = max([(child.Q, child) for child in startnode.children],key=lambda s:s[0])
        self.current_realnode = maxchild
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
                info['Data'].append( gamedata[i] )
                info['Data'].append( rightdata[i] )
            else:
                new_pos = len(gamedata) - 1
                break
        newtarget = np.reshape( rightdata[new_pos][0], [20,20] ).astype(np.int)
        play_to_the_end(newtarget , False, rightdata[new_pos:], info, nframes, eval_sess )
        return


def play_games(eval_sess,nframes,
               prob_blank_range = GLOBAL_PARAMETERS['blank_range'],
               height = GLOBAL_PARAMETERS['height'],
               width = GLOBAL_PARAMETERS['width'],

               ):
    Data = []
    n_game = 0
    total_score = 0.0
    print('Play games...:')
    while len(Data) < 6000:
        for prob_blank in prob_blank_range:
            sample = Create_sample(height, width, prob_blank)
            sample.add_pieces()
            target, solution = sample.T, sample.S
            rightdata = solve_game(target, solution)
            info = {'Data':[]}
            play_to_the_end(target, True, rightdata, info, nframes, eval_sess)
            Data.extend(info['Data'])
            n_game += 1
            total_score += info['first_score']
            print('game {}th to end with {} data'.format(n_game, len(info['Data'])), end='/', flush=True)

    avg_score = total_score / n_game
    return {'Data': Data, 'score': avg_score}


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
    data.append( ( np.reshape(target, [-1]), 1.0) )

    for y in range(0, row):
        for x in range(0, col):
            if target[y][x] == 1:
                target = deepcopy(target)
                (shape,pos) = solution[y][x]
                assert pos==0, 'top-left corner pos should be 0'
                for p in [0, 1, 2, 3]:
                    (x_, y_) = Goldenpositions[shape][p] - Goldenpositions[shape][0] + np.array([x, y])
                    assert withinrange(x_,y_,row,col),'x,y not withinrange'
                    assert target[y_][x_]==1, '(x,y) at target not to be 1'
                    target[y_][x_] = 0
                data.append(( np.reshape(target, [-1]) , 1.0))
    return data


def Main(dataset, savemodel, nframes, n_res_blocks,Tetris_filtering):

    learning_rate = 1e-4
    print( 'learning_rate: {}'.format(learning_rate) )
    print('Tetris_filtering:{}'.format(Tetris_filtering))

    if dataset:
        with open(dataset, 'rb') as dfile:
            total_data = pickle.load(dfile)
    else:
        total_data = []

    sess = build_neuralnetwork(20, 20, nframes, n_res_blocks, Tetris_filtering)
    with sess:
        saver = tf.train.Saver()
        if savemodel:
            saver.restore(sess, './{}.ckpt'.format(savemodel))
        else:
            sess.run(tf.global_variables_initializer())
        time_step = 0

        while True:
            # generate data, put into total data
            result = play_games(sess, nframes)
            print(time.strftime("%Y-%m-%d %H:%M:%S"),
                  ': Played a set of games ')
            newdata = result['Data']
            total_data.append(newdata)
            total_data = total_data[-GLOBAL_PARAMETERS['dataQ maxsize']:]
            with open('total_data.pickle', 'wb') as dfile:
                pickle.dump(total_data, dfile)
            print(time.strftime("%Y-%m-%d %H:%M:%S"),
                  ': Saved generated data into total_data.pickle which has {} datasets now'.format(len(total_data)))

            # training data
            newdata = np.array(newdata)
            shuffle_indices = np.random.permutation(np.arange(len(newdata)))
            shuffled_data = newdata[shuffle_indices]
            train_sample_index = max(int(0.8 * float(len(newdata))), len(newdata) - 2000)
            newtrain, test_data = shuffled_data[:train_sample_index], shuffled_data[train_sample_index:]
            training_data = []
            for dataset in total_data[:-1]:
                training_data.extend(dataset)
            training_data.extend(newtrain)

            #eval
            score = result['score']
            test_x, test_y = zip(*test_data)
            test_x = np.array(test_x)
            test_y = np.array(test_y)
            iter_orignal_loss = sess.run('MSError:0', feed_dict={
                'input_puzzles:0': test_x.astype(np.float32),
                'labels:0': test_y,
                'is_training:0': False
            })
            iter_minloss = iter_orignal_loss
            print(time.strftime("%Y-%m-%d %H:%M:%S"),
                  ': Generate {} data with score: {} and orignal_loss: {} '.format(len(newdata),score,iter_orignal_loss))
            print(time.strftime("%Y-%m-%d %H:%M:%S"),
                  ': which is divided into {} training data and {} test data'.format(len(newtrain), len(test_data)))
            print(time.strftime("%Y-%m-%d %H:%M:%S"),
                  ': Training data size:{}. Number of total datasets: {} '.format(len(training_data), len(total_data)) )

            #train
            epo = math.ceil( 320000 / len(training_data) )
            print('train for {} epoch'.format(epo))
            batches = data_batch_iter(training_data, 32, epo)
            epo_step = 0
            for inputdata, outputdata in batches:
                if epo_step%100 == 0:
                    trainloss = sess.run('MSError:0', feed_dict={
                        'input_puzzles:0': inputdata.astype(np.float32),
                        'labels:0': outputdata,
                        'is_training:0': False
                    })
                    print(time.strftime("%Y-%m-%d %H:%M:%S"),
                          ': total training step {}, iter step {}, loss {}'.format (time_step, epo_step, trainloss ))

                if epo_step % 2000 == 0:
                    testloss = sess.run('MSError:0', feed_dict={
                        'input_puzzles:0': test_x.astype(np.float32),
                        'labels:0': test_y,
                        'is_training:0': False
                    })
                    print(time.strftime("%Y-%m-%d %H:%M:%S"),
                          ': iter step %d, test loss %g' % (epo_step, testloss))
                    if testloss < iter_minloss:
                        iter_minloss = testloss
                        save_path = saver.save(sess, './{}.ckpt'.format(savemodel))
                        print(time.strftime("%Y-%m-%d %H:%M:%S"),
                              ': min iter loss model saved in file: {} '.format(save_path))

                sess.run('train_mini', feed_dict={
                    'input_puzzles:0': inputdata.astype(np.float32),
                    'labels:0': outputdata,
                    'is_training:0': True,
                    'learning_rate:0': learning_rate
                })
                time_step += 1
                epo_step += 1
                if epo_step == 4001:
                    break



            iter_final_loss = sess.run('MSError:0', feed_dict={
                        'input_puzzles:0': test_x.astype(np.float32),
                        'labels:0': test_y,
                        'is_training:0': False
                    })
            print(time.strftime("%Y-%m-%d %H:%M:%S"),
                  ': iter step {}, final iter test loss {}, iter min loss {}, iter original loss: {}'.format(epo_step, iter_final_loss, iter_minloss, iter_orignal_loss))
            if iter_final_loss > iter_minloss:
                saver.restore(sess, './{}.ckpt'.format(savemodel))
                print(time.strftime("%Y-%m-%d %H:%M:%S"),
                      ': Restore min iter model from {}'.format(savemodel))
            else:
                save_path = saver.save(sess, './{}.ckpt'.format(savemodel))
                print(time.strftime("%Y-%m-%d %H:%M:%S"),
                      ': final min iter loss model saved in file: {} '.format(save_path))



