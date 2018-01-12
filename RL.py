import numpy as np
import tensorflow as tf
from data_preprocessing import Goldenpositions
from copy import deepcopy

def build_neuralnetwork(height,width):
    x = tf.placeholder(tf.float32,shape= [None,width*height], name='input_puzzles')
    y_ = tf.placeholder(tf.float32, shape = [None], name='labels')
    is_training = tf.placeholder(tf.bool, name='is_training')

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



    x_puzzle = tf.reshape(x,[-1, height, width, 1])
    hidden = conv_block(x_puzzle, [3,3], 1, 256, is_training,"conv_block")

    for i in range(1,4):
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

    mse = tf.reduce_mean( tf.squared_difference(y,y_) )
    loss = tf.add(mse, lossL2, name='total_loss')

    #summary
    tf.summary.scalar('loss',loss)
    merged = tf.summary.merge_all()

    #train
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_step = tf.train.AdamOptimizer(1e-4).minimize(loss, name='train_mini')

    with tf.Session() as sess:
        return sess




class node(object):

    def __init__(self,state,table):
        self.table = table
        self.state = state
        self.expanded = False
        self.row = self.state.shape[0]
        self.col = self.state.shape[1]
        self.N = 0
        self.Q = 0.0
        self.W = 0.0


    def check_explore(self):
        """
        self.terminal indicates whether this state is terminal
        self.R is the immediate reward arriving this state
        :return: (terminal, reword)
         terminal: true if end, otherwise false
         R: if terminal: -1 or 1,
                  else not terminal, 0
        """
        self.expanded = True
        for y in range(0,self.row):
            for x in range(0,self.col):
                if self.state[y][x]==1:
                    self.fetch_children(x,y)
                    if len(self.children)>0:
                        np.random.shuffle( self.children )
                        self.terminal = False
                        self.R = 0.0
                        return
                    else:
                        self.terminal = True
                        self.R = -1.0
                        return

        self.terminal = True
        self.R = 1.0
        return

    def fetch_children(self,x,y):
        self.children = []
        for shape in range(1,20):
            available = True
            child_state = deepcopy(self.state)
            for p in [0, 1, 2, 3]:
                (x_, y_) = Goldenpositions[shape][p] - Goldenpositions[shape][0] + np.array([x, y])
                if self.withinrange(x_,y_,self.row,self.col) and child_state[y_][x_] == 1:
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
                    c_node = node(child_state,self.table)
                    # add in table
                    self.table[stateid] = c_node
                    #append child
                    self.children.append(c_node)

        return

    def withinrange(self,x, y, row, col):
        if x >= 0 and x < col and y >= 0 and y < row:
            return True
        else:
            return False





class simulation(object):
    def __init__(self,rootnode,L):
        self.rootnode = rootnode
        self.currentnode = rootnode
        self.path = [self.rootnode]
        self.L = L
        self.t = 0


    def run(self):

        while True:
            if not self.currentnode.expanded:
                self.currentnode.check_explore()
            if self.currentnode.terminal:
                self.backup(self.currentnode.R)
                break

            if self.t > 0 and (self.t%L==0):
                self.backup(networkvalue)

            self.currentnode = self.selectfrom(self.currentnode)
            self.path.append(self.currentnode)
            self.t += 1

        return

    def backup(self,v):
        for node in self.path:
            node.W += v
            node.N += 1
            node.Q = node.W/node.N
        return


    def selectfrom(self,node):
        values = []
        for child in node.children:
            v = child.Q + ( ( np.sqrt(2*node.N) ) / (1+child.N) )
            values.append((v,child))

        _,mchild = max(values, key=lambda s:s[0])
        return mchild





