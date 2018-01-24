# do not conduct MCTS, directly play the real move using child with the maximum predicted value


from data_preprocessing import Goldenpositions
from copy import deepcopy
import numpy as np

class Node(object):

    def __init__(self,state,table,cumR,uniR, sess):
        self.table = table
        self.state = state
        self.sess = sess
        self.cumR = cumR
        self.uniR = uniR
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
                        self.predict_childrenV()
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

    def predict_childrenV(self):
        predictions = self.sess.run('predictions:0', feed_dict={
            'input_puzzles:0': np.array([ np.reshape(child.state, [-1,]) for child in self.children]).astype(np.float32),
            'is_training:0': False})

        for i,child in enumerate(self.children):
            child.Q = predictions[i]
            child.W = predictions[i]
            child.N = 1.0



def withinrange(x, y, row, col):
    if x >= 0 and x < col and y >= 0 and y < row:
        return True
    else:
        return False


class Game(object):

    def __init__(self, target, n_search, L, sess):
        self.target = np.array(target)
        self.table = {}
        self.n_search = n_search
        self.L = L
        self.sess = sess

        self.current_realnode = Node(deepcopy(self.target),self.table, 0.0, 4/np.sum(self.target), self.sess)
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

                return gamedata, self.current_realnode.V, self.current_realnode.score
            else:
                self.play_one_move(self.current_realnode)


    def play_one_move(self,startnode):



        (maxQ,maxchild) = max([(child.Q, child) for child in startnode.children],key=lambda s:s[0])
        self.current_realnode = maxchild
        self.real_nodepath.append(self.current_realnode)

        return