import tensorflow as tf
import numpy as np
from copy import deepcopy
from data_preprocessing import Goldenpositions
import RL_supervised
from mysolution import Create_sample
import time


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
        self.sessP, self.sessV = self.sess
        self.expanded = False
        self.row = self.state.shape[0]
        self.col = self.state.shape[1]
        self.shape = self.state.shape

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
                    self.fetch_edges(x,y)
                    if len(self.edges)>0:
                        loc_frame = np.zeros(self.shape, dtype=np.int)
                        loc_frame[y][x] = 1
                        Ps = self.sessP.run('prob_predictions:0',
                                               feed_dict={'input_puzzles:0': np.reshape([self.state,loc_frame], [1, 2 * self.row * self.col]).astype(np.float32),
                                                          'is_training:0': False})
                        #Vs = self.sessP.run('value_predictions:0',
                        #                    feed_dict={'input_puzzles:0': np.reshape([self.state,loc_frame], [1, self.row * self.col]).astype(np.float32),
                        #                               'is_training:0': False})
                        self.terminal = False
                        self.Ps = Ps[0]
                        #self.V = Vs[0]
                        for edge in self.edges:
                            edge.P = self.Ps[edge.Prob_id]
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
                    ed = Edge(self, self.table[stateid], None, shape-1 )
                    self.edges.append(ed)
                else:
                    # create node
                    c_node = Node(child_state,self.table, self.cumR+self.uniR, self.uniR, self.sess)
                    # add in table
                    self.table[stateid] = c_node
                    #append edge
                    ed = Edge(self, c_node, None, shape-1)
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
                #self.backup(self.currentnode.V)
                #return
            #else:
            if self.currentnode.terminal:
                if self.currentnode.V > 0:
                    return True, [edge.end for edge in self.path]
                else:
                    self.backup(self.currentnode.score)
                    return False, None
            else:
                edge = self.selectfrom(self.currentnode)
                self.path.append(edge)
                self.currentnode = edge.end


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

    def __init__(self, target, n_search, nframes, sess, movetype):
        self.nframes = nframes
        self.target = np.array(target)
        self.table = {}
        self.n_search = n_search
        self.sess = sess
        self.movetype = movetype

        self.current_realnode = Node(deepcopy(self.target),self.table, 0.0, 4/np.sum(self.target), self.sess)
        self.table[self.current_realnode.state.tostring()] = self.current_realnode
        self.real_nodepath = [self.current_realnode]
        #self.search_Ps = []

        #self.current_realnode.check_explore()

    def play(self):

        while True:
            self.play_one_move(self.current_realnode)

            if self.current_realnode.terminal:
                #search_prob = np.ones(19, dtype=np.float32)/19
                #self.search_Ps.append(search_prob)
                #assert len(self.search_Ps)==len(self.real_nodepath), 'size of search prob not equal to real_nodepath'
                gamedata = []
                for i, node in enumerate(self.real_nodepath):
                    #if i == 0:
                    #    gamedata.append((np.reshape(node.state, [-1]), 1.0, self.search_Ps[i]))
                    #else:
                    gamedata.append((np.reshape(node.state, [-1]), self.current_realnode.V))#, self.search_Ps[i]))

                return gamedata, self.current_realnode.V, self.current_realnode.score



    def play_one_move(self,startnode):
        #noise = np.random.dirichlet([1.0 for i in range(len(startnode.edges))])
        #noise = np.ones(len(startnode.edges), dtype=np.float32) / len(startnode.edges)
        #for i,edge in enumerate(startnode.edges):
        #    edge.P = 0.75*edge.P + 0.25*noise[i]

        for i in range(0,self.n_search):
            simulation = Simulation(startnode, self.sess, self.nframes)
            sign, simulpath = simulation.run()
            if sign:
                self.real_nodepath.extend(simulpath)
                self.current_realnode = simulpath[-1]
                return

        if self.movetype == 'N':
            ( maxN, maxedge ) = max([(edge.N, edge) for edge in startnode.edges], key=lambda s:s[0])
            print('select max N: {}'.format(max([(edge.Q, edge.P, edge.N) for edge in startnode.edges], key=lambda s: s[2])))
            print('max Q: {}'.format(max([(edge.Q, edge.P, edge.N) for edge in startnode.edges], key=lambda s:s[0])))
            print('max P: {}'.format(max([(edge.Q, edge.P, edge.N) for edge in startnode.edges], key=lambda s: s[1])))
        if self.movetype == 'Q':
            (maxQ, maxedge) = max([(edge.Q, edge) for edge in startnode.edges], key=lambda s: s[0])
            print('select max Q: {}'.format(max([(edge.Q, edge.P, edge.N) for edge in startnode.edges], key=lambda s: s[0])))
            print('max N: {}'.format(max([(edge.Q, edge.P, edge.N) for edge in startnode.edges], key=lambda s: s[2])))
            print('max P: {}'.format(max([(edge.Q, edge.P, edge.N) for edge in startnode.edges], key=lambda s: s[1])))
        #search_prob = np.zeros(19, dtype=np.float32)
        #search_prob[maxedge.Prob_id] = 1.0
        #self.search_Ps.append( search_prob )

        self.current_realnode = maxedge.end
        self.real_nodepath.append(self.current_realnode)

        return

def Main(model):
    sessP = RL_supervised.build_Policynetwork(20, 20, 2, 10, False)
    with sessP:
        saver = tf.train.Saver()
        saver.restore(sessP, './{}.ckpt'.format(model))
        targets = []
        for i in range(100):
            sample = Create_sample(20, 20, 0.4)
            sample.add_pieces()
            target, solution = sample.T, sample.S
            targets.append(target)

        print('Play N...................')
        Nwin=0
        startt = time.time()
        for i,target in enumerate(targets):
            game = Game(target,100,None, [sessP,None], 'N')
            data, result, score = game.play()
            if result > 0:
                Nwin += 1
            print('game {}th, score {}'.format(i, score))
        endt = time.time()
        print('N results: win {}, time {}'.format(Nwin, endt - startt))

        print('Play Q...................')
        Qwin = 0
        startt = time.time()
        for i, target in enumerate(targets):
            game = Game(target,100,None,[sessP,None], 'Q')
            data,result,score = game.play()
            if result>0:
                Qwin += 1
            print('game {}th, score {}'.format(i,score))
        endt = time.time()
        print('Q results: win{}, time {}'.format(Qwin, endt-startt))



