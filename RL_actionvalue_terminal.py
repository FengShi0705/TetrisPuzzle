# This is the script for Approach 4 in reseach notes.
# This script use the simulation experience as training data and play each simulation to the end.
# However the problem is that this script use all the edges in the MCTS.
# Actually, we should only use the edges along the simulation paths.


import tensorflow as tf
import numpy as np
from copy import deepcopy
from data_preprocessing import Goldenpositions,data_batch_iter_three,data_batch_iter_six
from mysolution import Create_sample
import time
import pickle
import random


GLOBAL_PARAMETERS={
    'gamesize': 10,
    'blank_range':[0.4],
    'simulation per move': 100,
    'width':20,
    'height':20,
    'batchsize':32,
    'datasize': 300000
}



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
        action_taken = tf.placeholder(tf.float32, shape=[None,19], name='action_taken')
        legal_actions = tf.placeholder(tf.float32, shape=[None,19], name='legal_actions')
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
        hidden = conv_block(hidden, [1,1], 256, 2, is_training, 'conv_block_out')

        #fully connect to 256
        hidden_flat = tf.reshape(hidden, [-1, width*height*2])
        h_fc1 = tf.nn.relu( linear_fullyconnect(hidden_flat, width*height*2, 256, 'fully_connect_1') )

        #fully connect to 19 actions
        h_fc2 = linear_fullyconnect(h_fc1, 256, 19, 'fully_connect_2')
        ys = tf.sigmoid(h_fc2, name='predict_actionvalues')
        #y = tf.reshape(y, shape=[-1,], name='predictions')
        legal_values = tf.multiply(ys, legal_actions, name='legal_values')
        max_legal_values = tf.reduce_max(legal_values, axis=1, name='max_legal_action')
        ys = tf.multiply(ys, action_taken)
        y = tf.reduce_sum(ys, axis=1, name='actionvalue_Taken')

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

    return myGraph


class Edge(object):
    def __init__(self,start,end, action_id):
        self.start = start
        self.end = end
        #self.N = 1
        self.action_id = action_id



class Node(object):

    def __init__(self,state,table,remain_steps, sess, alledges):
        self.table = table
        self.alledges = alledges
        self.state = state
        self.remain_steps = remain_steps
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
        self.edges = []

        for y in range(0,self.row):
            for x in range(0,self.col):
                if self.state[y][x]==1:
                    self.fetch_edges(x,y)
                    if len(self.edges)>0:
                        self.terminal = False
                        loc_frame = np.zeros(self.state.shape, dtype=np.int)
                        loc_frame[y][x] = 1
                        self.feed_input = np.reshape([self.state,loc_frame], [2 * self.row * self.col])
                        self.Qs = self.sess.run('predict_actionvalues:0',
                                           feed_dict={'input_puzzles:0': np.reshape(self.feed_input, [1, 2 * self.row * self.col]).astype(np.float32),
                                                      'is_training:0': False})[0]
                        for edge in self.edges:
                            edge.Q = self.Qs[edge.action_id]
                            edge.N = 0
                            #edge.W = edge.Q

                        self.V = max([edge.Q for edge in self.edges])
                        return
                    else:
                        self.terminal = True
                        self.V = 0.0
                        return

        self.terminal = True
        self.V = 0.0
        return

    def fetch_edges(self,x,y):
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
                    ed = Edge(self, self.table[stateid], shape-1 )
                    self.edges.append(ed)
                else:
                    # create node
                    c_node = Node(child_state, self.table, self.remain_steps-1, self.sess, self.alledges)
                    # add in table
                    self.table[stateid] = c_node
                    #append edge
                    ed = Edge(self, c_node, shape-1)
                    self.edges.append(ed)

                self.alledges.append(ed)

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

    def __init__(self,rootnode, sess, toEND = False, updatemax = True ):
        self.toEnd = toEND
        self.updatemax = updatemax
        self.currentnode = rootnode
        self.sess = sess
        self.path = []



    def run(self):

        while True:
            if not self.currentnode.expanded:
                self.currentnode.check_explore()
                # self.backup(self.currentnode.V, self.currentnode.remain_steps)
                #if not self.toEnd:
                #    return
            #else:
            if self.currentnode.terminal:
                self.backup(self.currentnode.V, self.currentnode.remain_steps)
                return
            else:
                edge = self.selectfrom(self.currentnode)
                self.path.append(edge)
                self.currentnode = edge.end


    def backup(self,v, sp):
        assert v == 0.0, 'terminal V should be 0.0, but not'
        for edge in reversed(self.path):
            step = edge.start.remain_steps
            newQ = ((step - sp)/step) + (sp/step * v)
            #edge.W += ((step - sp) / step) + (sp / step * v)
            #edge.N += 1
            #edge.Q = edge.W / edge.N
            #if self.updatemax:
            #    if len(edge.start.edges) > 1:
            #        if edge.Q < max([ed.Q for ed in edge.start.edges if ed is not edge]):
            #            break

            if edge.N == 0 :
                edge.N += 1
                edge.Q = newQ
            else:
                edge.N += 1
                if newQ > edge.Q:
                    edge.Q = newQ
        return




    def selectfrom(self,node):
        # there is no probability, soly depend on Q and N
        sum_N = np.sum([ edge.N for edge in node.edges ])
        value_max = (float('-inf'), None)
        for edge in node.edges:
            v = edge.Q + ( ( 0.5 * np.sqrt(sum_N) ) / (1 + edge.N) )
            if v > value_max[0]:
                value_max = (v, edge)

        return value_max[1]


class Game(object):

    def __init__(self, target, n_search, sess, proportional, toEND, updatemax):
        self.proportional = proportional
        self.toEND = toEND
        self.updatemax = updatemax
        self.Data = []
        self.target = np.array(target)
        self.table = {}
        self.alledges = []
        self.n_search = n_search
        self.sess = sess

        self.current_realnode = Node(deepcopy(self.target),self.table, np.sum(self.target)/4, self.sess, self.alledges)
        self.table[self.current_realnode.state.tostring()] = self.current_realnode
        self.real_nodepath = [self.current_realnode]

        self.current_realnode.check_explore()


    def play(self):

        while True:
            self.play_one_move(self.current_realnode)

            if self.current_realnode.terminal:
                for edge in self.alledges:
                    if edge.end.expanded:
                        self.store_trains(edge)
                return self.Data, self.current_realnode.V, self.current_realnode.remain_steps


    def store_trains(self,edge):
        startn, endn = edge.start, edge.end
        action_taken = np.zeros(19, dtype=np.float32)
        action_taken[edge.action_id] = 1.0
        legal_actions = np.zeros(19, dtype=np.float32)
        for endedge in endn.edges:
            legal_actions[endedge.action_id] = 1.0

        if endn.terminal:
            self.Data.append( (startn.feed_input,
                                    action_taken,
                                    1.0 / startn.remain_steps,
                                    0.0,
                                    np.zeros(2 * endn.row * endn.col, dtype=np.int ),
                                    np.zeros(19, dtype=np.float32)) )
        else:
            self.Data.append( (startn.feed_input,
                               action_taken,
                               1.0/startn.remain_steps,
                               1- (1.0/startn.remain_steps),
                               endn.feed_input,
                               legal_actions) )

        return



    def play_one_move(self,startnode):

        for i in range(0,self.n_search):
            simulation = Simulation(startnode, self.sess, self.toEND, self.updatemax)
            simulation.run()

        if self.proportional:
            sl_edge = self.select_proportional_N(startnode)
        else:
            # soly on Q
            (maxQ,sl_edge) = max([(edge.Q, edge) for edge in startnode.edges], key=lambda s:s[0])

        self.current_realnode = sl_edge.end
        self.real_nodepath.append(self.current_realnode)
        return

    def select_proportional_N(self,node):
        ns = [edge.N for edge in node.edges]
        sum_n = sum(ns)
        track = 0
        prob = []
        for n in ns:
            track += n/sum_n
            prob.append(track)
        randv = np.random.uniform()
        for i,v in enumerate(prob):
            if randv < v:
                eid = i
                break
        return node.edges[eid]


def generate_targets(size, name,
                     prob_blank_range=GLOBAL_PARAMETERS['blank_range'],
                     height=GLOBAL_PARAMETERS['height'],
                     width=GLOBAL_PARAMETERS['width']
                     ):
    targets = []
    for i in range(size):
        for prob_blank in prob_blank_range:
            sample = Create_sample(height, width, prob_blank)
            sample.add_pieces()
            target = sample.T
            targets.append(target)

    with open('{}_{}.pickle'.format(name,size), 'wb') as dfile:
        pickle.dump(targets, dfile)
    return




class Train:
    def __init__(self, model, targets_set, dataset):
        self.dataset = dataset
        with open(self.dataset, 'rb') as dfile:
            self.total_data = pickle.load(dfile)
        self.model = model
        self.graph = build_neuralnetwork(20, 20, 2, 10, False)
        self.sess = tf.Session(graph=self.graph)
        self.sess_target = tf.Session(graph=self.graph)
        with open(targets_set, 'rb') as dfile:
            self.targets = pickle.load(dfile)

        with self.graph.as_default():
            self.saver = tf.train.Saver()
            self.saver.restore(self.sess, './{}.ckpt'.format(self.model))
            self.saver.restore(self.sess_target, './{}.ckpt'.format(self.model))

    def play_games(self):
        #np.random.shuffle(self.targets)

        Data = []
        n_game = 0
        total_score = 0.0
        print('Play games...:')
        #for target in self.targets[:GLOBAL_PARAMETERS['gamesize']]:
        while True:
            targi = random.randint(0,999)
            game = Game(self.targets[targi], GLOBAL_PARAMETERS['simulation per move'], self.sess, False, True, True)
            gamedata, _, score = game.play()
            Data.extend(gamedata)
            print('game {}th, data {}, score {}, proportional False'.format(targi, len(gamedata), score), end='/',flush=True)
            n_game += 1
            total_score += score
            # if len(Data) > 10000 :
            break

        self.total_data.extend(Data)
        self.total_data = self.total_data[-GLOBAL_PARAMETERS['datasize']:]
        with open(self.dataset, 'wb') as dfile:
            pickle.dump(self.total_data, dfile)
        avg_score = total_score / n_game
        print(time.strftime("%Y-%m-%d %H:%M:%S"),
              'Saved {} data with {} into {}, which has {} total data'.format(len(Data),
                                                                              avg_score,
                                                                              self.dataset,
                                                                              len(self.total_data))
              )
        self.trainsteps = len(Data)
        return


    def run(self):

        self.time_step = 0

        while True:
            self.play_games()
            self.train()

        return

    def train(self):
        self.epo_step = 0
        batches = data_batch_iter_six(self.total_data, 32, 1)
        for startstates, actions, rewards, discounts, endstates, legal_acts in batches:
            values = self.get_learning_target(rewards, discounts, endstates, legal_acts)

            if self.epo_step % 100 == 0:
                self.check_step(startstates,actions,values)

            if self.time_step % 3000 == 0:
                self.check_step(startstates, actions, values)
                self.update_save()
                values = self.get_learning_target(rewards, discounts, endstates, legal_acts)
                self.check_step(startstates, actions, values)

            self.sess.run('train_mini', feed_dict={
                'input_puzzles:0': startstates.astype(np.float32),
                'action_taken:0': actions,
                'labels:0': values,
                'is_training:0': True,
                'learning_rate:0': 1e-4
            })
            self.time_step += 1
            self.epo_step += 1
            #if self.epo_step > self.trainsteps:
            #    return


    def update_save(self):
        path = self.saver.save(self.sess, './{}.ckpt'.format(self.model))
        self.saver.restore(self.sess_target, './{}.ckpt'.format(self.model))
        print('total time step {}, epo step {}, Save and restore {}'.format(self.time_step, self.epo_step, path))
        return

    def check_step(self, states, actions, values):
        train_mse, preV = self.sess.run(['MSError:0', 'actionvalue_Taken:0'], feed_dict={
            'input_puzzles:0': states.astype(np.float32),
            'action_taken:0': actions,
            'labels:0': values,
            'is_training:0': False
        })
        print(time.strftime("%Y-%m-%d %H:%M:%S"),
              ': total step {}, epo step {}, train mse {}'.format(self.time_step, self.epo_step, train_mse))
        print('target values: {}'.format(values))
        print('current values: {}'.format(preV))

    def get_learning_target(self,rewards, discounts, endstates, legal_acts):
        maxvalues = self.sess_target.run('max_legal_action:0', feed_dict={
            'input_puzzles:0': endstates.astype(np.float32),
            'legal_actions:0': legal_acts,
            'is_training:0': False
        })
        values = np.add(rewards, np.multiply(discounts, maxvalues))
        return values




