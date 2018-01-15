# ###########################################
# This script applys reinforcement learning and MCTS in tiling problems.

# In MCTS,
# 1. Pure -1 and +1 rewards during simulation have two disadvantages:
#          (1): if all the simulation in a search have reward of (-1), the next real move will benefit nothing from the search
#          (2): multiple simulations may end up in the same terminal state, which wastes the search resources
#    In order to tackle this two disadvantages, we take into consideration the depth of the terminal state for the reward.
# 2. We also use backup the state-value every L step, this make use of the training network to guild the search.
##############################################
import numpy as np
import tensorflow as tf
from data_preprocessing import Goldenpositions
from copy import deepcopy
import utils
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image, ImageDraw

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


class Edge(object):
    def __init__(self,start,end):
        self.start = start
        self.end = end
        self.N = 0
        self.Q = 0.0
        self.W = 0.0

class Node(object):

    def __init__(self,state,table,cumR,uniR):
        self.table = table
        self.state = state
        self.cumR = cumR
        self.uniR = uniR
        self.expanded = False
        self.row = self.state.shape[0]
        self.col = self.state.shape[1]


    def check_explore(self):
        """
        self.terminal indicates whether this state is terminal
        self.R is the immediate reward arriving this state
         terminal: true if end, otherwise false
         R: if terminal: -1 or 1,
                  else not terminal, 0
        """
        self.expanded = True
        for y in range(0,self.row):
            for x in range(0,self.col):
                if self.state[y][x]==1:
                    self.fetch_edges(x,y)
                    if len(self.edges)>0:
                        self.terminal = False
                        self.Qv = 0.0
                        return
                    else:
                        self.terminal = True
                        self.Qv = -1.0 + self.cumR
                        self.V = -1.0
                        return

        self.terminal = True
        self.Qv = 0.0 + 1.0
        self.V = 1.0
        return

    def fetch_edges(self,x,y):
        self.edges = []
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
                    #self.children.append( self.table[stateid] )
                    ed = Edge(self, self.table[stateid])
                    self.edges.append(ed)
                else:
                    # create child
                    c_node = Node(child_state,self.table, self.cumR+self.uniR, self.uniR)
                    # add in table
                    self.table[stateid] = c_node
                    #append child
                    #self.children.append(c_node)
                    ed = Edge(self,c_node)
                    self.edges.append(ed)

        return

    def withinrange(self,x, y, row, col):
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

    def __init__(self,rootnode,L,sess):
        self.currentnode = rootnode
        self.sess = sess
        self.path = []
        self.L = L
        self.t = 0


    def run(self):

        while True:
            if not self.currentnode.expanded:
                self.currentnode.check_explore()
            if self.currentnode.terminal:
                self.backup(self.currentnode.Qv)
                break

            if self.t > 0 and (self.t%self.L==0):
                #predict_value = self.sess.run('predictions:0',
                #                              feed_dict={'input_puzzles:0':np.reshape(self.currentnode.state,[1,-1]).astype(np.float32),
                #                                         'is_training:0': False}
                #                              )
                #self.backup(predict_value[0])
                self.backup(self.currentnode.Qv)
                for edge in self.currentnode.edges:
                    edge.N = 0
                    edge.Q = 0.0
                    edge.W = 0.0

            edge = self.selectfrom(self.currentnode)
            self.path.append(edge)
            self.currentnode = edge.end
            self.t += 1

        return

    def backup(self,v):
        for edge in self.path:
            edge.W += v
            edge.N += 1
            edge.Q = edge.W/edge.N
        return


    def selectfrom(self,node):
        sum_N = np.sum([ edge.N for edge in node.edges ])
        value_max = (-100.0, None)
        for edge in node.edges:
            v = edge.Q + ( ( np.sqrt(2*sum_N) ) / (1+edge.N) )
            if v > value_max[0]:
                value_max = (v, edge)

        return value_max[1]


class Game(object):

    def __init__(self, target, n_search, L, sess):
        self.target = np.array(target)
        self.table = {}
        self.n_search = n_search
        self.L = L
        self.sess = sess

        self.current_realnode = Node(deepcopy(self.target),self.table, 0.0, 4/np.sum(self.target))
        self.table[self.current_realnode.state.tostring()] = self.current_realnode
        self.real_nodepath = [self.current_realnode]

    def play(self):

        while True:
            if not self.current_realnode.expanded:
                self.current_realnode.check_explore()

            if self.current_realnode.terminal:
                gamedata = []
                for i, node in enumerate(self.real_nodepath):
                    if i==0:
                        gamedata.append( (np.reshape(node.state, [-1]), 1.0) )
                    else:
                        gamedata.append( (np.reshape(node.state, [-1]), self.current_realnode.V) )

                return gamedata
            else:
                self.play_one_move(self.current_realnode)


    def play_one_move(self,startnode):
        for i in range(0,self.n_search):
            simulation = Simulation(startnode,self.L,self.sess)
            simulation.run()


        (maxQ,maxedge) = max([(edge.Q, edge) for edge in startnode.edges],key=lambda s:s[0])
        self.current_realnode = maxedge.end
        self.real_nodepath.append(self.current_realnode)

        return


def visualisation(target, solution):
    """
    Displays the target vs the solution
    :param target: target shape
    :param solution: student's solution
    """
    Ty_len = len(target)
    Tx_len = len(target[0])
    Sy_len = len(solution)
    Sx_len = len(solution[0])

    fig, (ax1, ax2) = plt.subplots(1, 2)  # Create figure and axes
    im = Image.new('RGB', (Tx_len, Ty_len), (255, 255, 255))  # white background-image
    dr = ImageDraw.Draw(im)
    ax1.imshow(im)  # Display the background-image
    ax2.imshow(im)

    # -------------------- Target Display ----------------------
    for y in range(Ty_len):
        row = target[y]
        for x in range(Tx_len):
            if row[x] == 1:
                ax1.add_patch(patches.Rectangle((x, y), 0.88, 0.88, color='b'))  # draw a block
    ax1.set_title('The Display of Task')
    ax1.set_xlim([-1, Tx_len + 1])
    ax1.set_ylim([-1, Ty_len + 1])
    ax1.invert_yaxis()

    # --------------- Solution Display ----------------------
    for y in range(Sy_len):
        row = solution[y]
        for x in range(Sx_len):
            if row[x] == 1:
                ax2.add_patch(patches.Rectangle((x, y), 0.88, 0.88, color='b'))  # draw a block
    ax2.set_title('The Display of Task')
    ax2.set_xlim([-1, Sx_len + 1])
    ax2.set_ylim([-1, Sy_len + 1])
    ax2.invert_yaxis()


    plt.show()

if __name__ == "__main__":
    sess = build_neuralnetwork(20, 20)
    sess.run(tf.global_variables_initializer())
    r1 = 0
    r2 = 0
    for i in range(0,100):
        print('--------------')
        target = utils.generate_target(width=20, height=20, density=0.6)
        #target=[[0, 1, 1, 0, 0, 0, 1, 1, 1, 1], [0, 0, 1, 1, 1, 1, 1, 1, 0, 1], [0, 0, 0, 0, 1, 1, 1, 0, 1, 1], [1, 1, 1, 1, 0, 1, 0, 1, 1, 1], [0, 1, 1, 1, 0, 1, 0, 1, 0, 1], [0, 1, 1, 1, 1, 1, 1, 1, 0, 0], [1, 1, 0, 0, 1, 1, 0, 1, 0, 0], [1, 0, 0, 0, 1, 1, 1, 0, 0, 1], [1, 0, 0, 1, 1, 1, 0, 1, 1, 1], [0, 1, 1, 1, 1, 0, 0, 0, 0, 0]]


        game1 = Game(target,100,5,sess)
        game2 = RLnode.Game(target,100,5,sess)
        game1data = game1.play()
        game2data = game2.play()
        if len(game1data)>len(game2data):
            r1 += 1
        if len(game1data)<len(game2data):
            r2 += 1

        print('backup+init edge wins: {}'.format(r1))
        print('backup node wins: {}'.format(r2))

        """for step in game1data:
            visualisation(target,np.reshape(step[0],[10,10]))
        for step in game2data:
            visualisation(target, np.reshape(step[0], [10, 10]))"""

    """target = np.array([[1, 1, 1, 1, 0, 0, 0, 1, 1, 1],
        [1, 1, 1, 1, 0, 1, 0, 1, 1, 1],
        [1, 1, 0, 1, 0, 1, 1, 1, 0, 0],
        [1, 1, 1, 0, 1, 1, 0, 0, 0, 0],
        [0, 0, 1, 1, 0, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 1, 1, 0, 1, 0, 0],
        [1, 0, 1, 1, 1, 1, 0, 1, 1, 1],
        [1, 1, 1, 0, 0, 0, 0, 0, 1, 1],
        [1, 0, 1, 0, 1, 0, 0, 0, 1, 1],
        [0, 1, 1, 1, 1, 1, 1, 0, 0, 0]])"""





