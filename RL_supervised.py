from copy import deepcopy
from data_preprocessing import Goldenpositions
import numpy as np
from mysolution import Create_sample
import pickle
import multiprocessing

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
        sum_N = np.sum([ edge.N for edge in node.edges ])
        value_max = (float('-inf'), None)
        for edge in node.edges:
            v = edge.Q + ( ( edge.P * np.sqrt(2*sum_N) ) / (1 + edge.N) )
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
                search_prob = np.ones(19, dtype=np.float32)/19
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
        search_prob[maxedge.Prob_id] = 1.0
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

    with open('{}'.format('supervised_PV_data_{}.pickle'.format(name)), 'wb') as dfile:
        pickle.dump(Data, dfile)
    scores=np.array(scores)
    return {'mean score': scores.mean(), 'std':scores.std()}


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
    search_prob = np.ones(19, dtype=np.float32) / 19
    loc_frame = np.zeros([row,col],dtype=np.int)
    loc_frame[row-1][col-1] = 1
    data.append((np.reshape(deepcopy([target,loc_frame]), [-1]), 1.0, search_prob))

    return data


def Main():
    pool = multiprocessing.Pool(8)
    results = pool.starmap(play_games, [(None, 1, i,) for i in range(8)])
    for result in results:
        print(result)
    print('Finish Done')

