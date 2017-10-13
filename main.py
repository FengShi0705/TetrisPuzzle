import tensorflow as tf
from data_preprocessing import processPuzzle,indexto_shape_pos,shapePosto_index,one_to_threeotherpositions,processSquare,Half_width,Half_height,WindowSize
import numpy as np
from heapq import heappush, heappop
import matplotlib.pyplot as plt
import utils
from copy import deepcopy
import itertools


Count = itertools.count()

def Tetris(target,fig,ax):
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
        #probability = prob.eval(feed_dict={
        #    x:batch_x, keep_prob:1.0
        #})
        tile = Tiling(target, sess, prob, keep_prob, x, fig, ax)
        tile.fill()

    return tile.M, tile.S


class Tiling(object):

    def __init__(self, target, sess, prob_eval, keep_prob, input_x, fig, ax):
        self.target = np.array(target, dtype=np.float32)
        self.realup_T = deepcopy(self.target)
        self.row = self.target.shape[0]
        self.col = self.target.shape[1]
        self.M = np.empty([self.row, self.col], object)
        self.S = np.empty([self.row, self.col], object)
        self.fig = fig
        self.ax = ax

        self.session = sess
        self.prob_eval = prob_eval
        self.keep_prob = keep_prob
        self.inputx = input_x

        #preprocess data
        inputdata=[]
        processPuzzle(self.target,None,inputdata)
        inputdata = np.array(inputdata)

        #probability
        all_probability = self.session.run(self.prob_eval, feed_dict={self.inputx:inputdata, self.keep_prob:1.0} )
        NodeC.Node_list = []
        NodeC.Node_matrix = np.empty([self.row, self.col], dtype=object)
        NodeC.Shape = (self.row, self.col)
        NodeC.realup_T = self.realup_T
        NodeC.session = sess
        NodeC.prob_eval = prob_eval
        NodeC.keep_prob = keep_prob
        NodeC.inputx = input_x
        NodeC.M = self.M
        NodeC.T = self.target
        # create node objects and save into Node_matrix
        Pnum = 0
        for x in range(0, self.col):
            for y in range(0, self.row):
                node = NodeC(all_probability[Pnum], x, y)
                Pnum += 1
                NodeC.Node_matrix[y][x] = node

        # calculate initial score
        for rnodes in NodeC.Node_matrix:
            for node in rnodes:
                node.update_score()


    def fill(self):

        # show target
        #plt.ion()
        #self.fig = plt.figure()
        #self.ax1,self.ax2 = utils.showtarget(self.target, self.fig)

        self.Pid = 1
        while len(NodeC.Node_list) > 0:
            (neg_score, c_, node) = heappop(NodeC.Node_list)
            if (self.M[node.y][node.x] != None) or (node.score != (-neg_score)):
                continue

            #### ignore the rest fill (0,0) into 0
            if self.target[node.y][node.x] == 0.0 and node.shapeid == 0:
                break


            self.updateAll(node)
            print('fill node: x({}), y({}), shape({}), pos({}), and the other corresponding three nodes'.format(node.x,
                                                                                                                node.y,
                                                                                                                node.shapeid,
                                                                                                                node.pos))

            #if self.updateAll(node):
            #    if node.shapeid !=0 :
            #        self.Pid += 1
            #    print('fill node: x({}), y({}), shape({}), pos({}), and the other corresponding three nodes'.format(node.x, node.y, node.shapeid, node.pos))
            #else:
            #    node.update_score()

        # finalize figure
        #plt.ioff()
        #plt.show()
        # deal the rest 0
        for x in range(0,self.col):
            for y in range(0,self.row):
                if self.M[y][x]==None:
                    self.M[y][x] = (0, 0)
                    self.S[y][x] = (0, 0)

        return

    def updateAll(self,node):
        #sign = True

        #for (x_, y_) in positions:
        #    if (not withinrange(x_, y_, self.row, self.col)) or self.M[y_][x_] != None:
        #        sign = False
        #        return sign
        #assert sign == True, 'Sign should be true to update'

        positions = node.positions + [(node.x, node.y)]
        info = node.info + [[(node.x, node.y), node.shapeid, node.pos]]

        if node.shapeid == 0:
            pid = 0
        else:
            pid = self.Pid
            self.Pid += 1

        # update M, S, realup_T, Pid
        for inf in info:
            # update
            (x, y), shape, pos = inf
            assert withinrange(x,y, self.row, self.col), 'node not withinrange'
            assert self.M[y][x] == None, 'Filling conflict piece'
            self.M[y][x] = (shape, pid)
            assert self.S[y][x] == None, 'Filling conflict piece'
            self.S[y][x] = (shape, pos)
            self.realup_T[y][x] = 0.0


        # colornodes for redraw
        utils.update_ax(self.fig, positions, self.ax, pid)

        #update node prob and score
        allx,ally = zip(*positions)
        x_prob_low = min(allx) - Half_width
        x_prob_up = max(allx) + Half_width
        y_prob_low = min(ally) - Half_height
        y_prob_up = max(ally) + Half_height
        self.update_range_probability_score(x_prob_low, x_prob_up, y_prob_low, y_prob_up)
        #calc node score
        self.calc_range_score(x_prob_low, x_prob_up, y_prob_low, y_prob_up)

        return

    def update_range_probability_score(self, xl, xu, yl, yu): #ok
        for upx in range( max(0, xl ), min(self.col, xu  + 1)):
            for upy in range( max(0, yl), min(self.row, yu  + 1)):
                if self.M[upy][upx] == None:
                    upnode = NodeC.Node_matrix[upy][upx]
                    upnode.update_prob()

        for upx in range(max(0, xl), min(self.col, xu + 1)):
            for upy in range(max(0, yl), min(self.row, yu + 1)):
                if self.M[upy][upx] == None:
                    upnode = NodeC.Node_matrix[upy][upx]
                    upnode.update_score()

    def calc_range_score(self, xl, xu, yl, yu): #ok
        def calc_node_score(x,y):
            if self.M[y][x] == None:
                node = NodeC.Node_matrix[y][x]
                node.update_score(pop=False)
            return

        for x in range( max(xl-3, 0), min(xu+4,self.col) ):
            if x<xl:
                for y in range(max( yl - (x - (xl - 3)), 0), min( yu + (x - (xl - 3)) + 1, self.row )):
                    calc_node_score(x, y)
            elif x<=xu:
                for y in range( max(yl-3, 0) , yl ):
                    calc_node_score(x, y)
                for y in range(yu + 1, min(yu + 4, self.row)):
                    calc_node_score(x, y)
            else:
                for y in range( max(yl-(xu+3-x),0), min( yu + (xu + 3 - x) + 1, self.row )  ):
                    calc_node_score(x, y)

        return

class NodeC(object):
    Node_list=[]
    Node_matrix = None
    Shape = None
    realup_T = None
    session = None
    prob_eval = None
    keep_prob = None
    inputx = None
    M = None
    T = None


    def __init__(self, probability, x, y):
        self.probability=probability
        self.x=x
        self.y=y

        self.generate_probheap()

    def generate_probheap(self):
        if NodeC.T[self.y][self.x] == 0.0: # if a square is empty, lower the probability of 0 class
            self.probability[76] = min(self.probability[76], 1.01*np.max(self.probability[:-1]) )

        self.sort_prob = []
        for i, p in enumerate(self.probability):
            shapeid = int(i / 4) + 1
            pos = i % 4
            if shapeid == 20:
                shapeid = 0
            self.sort_prob.append((p, (shapeid, pos)))

        self.sort_prob.sort(key=lambda s:s[0])


    def update_prob(self):
        ipt_x = processSquare(NodeC.realup_T, self.x,self.y)
        ipt_x = ipt_x.reshape(-1, WindowSize[0]*WindowSize[1])
        outputy = NodeC.session.run(NodeC.prob_eval, feed_dict={NodeC.inputx:ipt_x, NodeC.keep_prob:1.0} )
        self.probability = outputy[0]
        self.generate_probheap()


    def update_score(self, pop = True):
        if pop:
            (p, (shapeid,pos))=self.sort_prob.pop()
            self.p = p
            self.shapeid=shapeid
            self.pos=pos
            self.info, self.positions = one_to_threeotherpositions(self.x, self.y, self.shapeid, self.pos)

        self.score = self.p
        sign = True
        for inf in self.info:
            (x_, y_), shape_, pos_ = inf
            if (not withinrange(x_,y_, NodeC.Shape[0], NodeC.Shape[1])) or (NodeC.M[y_][x_] != None ):
                sign = False
                break
            else:
                index_ = shapePosto_index(shape_, pos_)
                self.score += NodeC.Node_matrix[y_][x_].probability[index_]

        if sign: # update score until a valid one
            heappush(NodeC.Node_list, (-self.score, next(Count), self))
            return
        else:
            self.update_score()
            return


def withinrange(x, y, row, col):
    if x >= 0 and x < col and y >= 0 and y < row:
        return True
    else:
        return False
























