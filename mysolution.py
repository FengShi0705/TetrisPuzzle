import numpy as np
import random
import json
from check import check
import itertools
import operator

# most top(priority) and most left is [0,0],
#top->down, left->right
Goldenpositions = {
    1: np.array([[0,0],[1, 0], [0, 1], [1, 1]]),
    2: np.array([[0,0],[0, 1], [0, 2], [0, 3]]),
    3: np.array([[0,0],[1, 0], [2, 0], [3, 0]]),
    4: np.array([[0,0],[0, 1], [0, 2], [1, 2]]),
    5: np.array([[0,0],[-2, 1], [-1, 1], [0, 1]]),
    6: np.array([[0,0],[1, 0], [1, 1], [1, 2]]),
    7: np.array([[0,0],[1, 0], [2, 0], [0, 1]]),
    8: np.array([[0,0],[0, 1], [-1, 2], [0, 2]]),
    9: np.array([[0,0],[1, 0], [2, 0], [2, 1]]),
    10: np.array([[0,0],[1, 0], [0, 1], [0, 2]]),
    11: np.array([[0,0],[0, 1], [1, 1], [2, 1]]),
    12: np.array([[0,0],[0, 1], [1, 1], [0, 2]]),
    13: np.array([[0,0],[-1, 1], [0, 1], [1, 1]]),
    14: np.array([[0,0],[-1, 1], [0, 1], [0, 2]]),
    15: np.array([[0,0],[1, 0], [2, 0], [1, 1]]),
    16: np.array([[0,0],[1, 0], [-1, 1], [0, 1]]),
    17: np.array([[0,0],[0, 1], [1, 1], [1, 2]]),
    18: np.array([[0,0],[1, 0], [1, 1], [2, 1]]),
    19: np.array([[0,0],[-1, 1], [0, 1], [-1, 2]])
}

class fill_piece(object):
    """
    fill a piece in the frame based on the last unused square
    """
    tysq_list = []
    for shape in np.arange(1, 20, 1, dtype=int):
        for pos in [0, 1, 2, 3]:
            tysq_list.append((shape, pos))

    def __init__(self, T, S, M, unused, PiD):
        self.typesquares = fill_piece.tysq_list[:]
        random.shuffle(self.typesquares)
        self.T = T
        self.M = M
        self.S = S
        self.unused = unused
        self.PiD = PiD
        (self.x, self.y) = self.unused[-1]

    def prob(self,probability):
        if random.uniform(0, 1)<=probability:
            return True
        else:
            return False

    def fill(self):
        try:
            (shape, pos)=self.typesquares.pop()
        except:
            return False
        info, positions = self.one_to_threepositions(shape,pos)
        if len( list( set(positions).difference(set(self.unused)) ) )>0:
            return self.fill()
        else:
            for inf in info:
                self.M[ inf[0][1] ][ inf[0][0] ]= (shape, self.PiD)
                self.S[ inf[0][1] ][ inf[0][0] ]= (shape, inf[2])
                self.unused.remove(inf[0])
            #if self.prob(0.1):
            #    removeitem = random.choice(positions)
            #    positions.remove(removeitem)
            for tposition in positions:
                self.T[ tposition[1] ][ tposition[0] ]=1
            return True


    def one_to_threepositions(self,shape, pos):
        info = [[(self.x,self.y),shape,pos]]
        positions=[(self.x,self.y)]
        for p in [0, 1, 2, 3]:
            if p != pos:
                (x_, y_) = Goldenpositions[shape][p] - Goldenpositions[shape][pos] + np.array([self.x, self.y])
                positions.append((x_, y_))
                info.append([(x_, y_),shape,p])
        return info,positions




class Create_sample(object):
    """
    create a sample of puzzle to be filled with Tetris pieces
    """
    def __init__(self,rows,columns,percentage):
        """
        initialization a puzzle sample
        :param rows: number of rows of the puzzle
        :param columns: number of columns
        :param percentage: predifined percentage of the puzzle to be blanked
        """
        self.total_squars=rows*columns
        self.tofill = self.total_squars*(1-percentage)
        self.PiD = 1

        self.T=np.zeros([rows,columns],int)
        self.S = np.zeros([rows, columns,2],int)
        self.M=np.zeros([rows,columns,2],int)
        self.unused=[]
        for x in range(0,columns):
            for y in range(0,rows):
                self.unused.append((x,y))
        random.shuffle(self.unused)
        self.unused=self.unused[0:int(self.tofill)]

    def add_pieces(self):
        while len(self.unused)>0:
            piece=fill_piece(self.T,self.S,self.M,self.unused,self.PiD)
            if piece.fill():
                self.PiD += 1
            else:
                self.unused.pop()
        return None

def check_ary_contains_ary(array1,array2):
    """
    check if rows in array1 are all contained in array2
    :param array1: numpy array
    :param array2: numpy array
    :return: if all the rows in array1 are contained in array2, return True. Otherwise, return False
    """
    list1 = array1.tolist()
    list2 = array2.tolist()
    sign=True
    for n in list1:
        if n not in list2:
            sign = False

    return sign

def threetofour(x1,y1,x2,y2,x3,y3):
    allpossibility=[]

    three=[(x1,y1),(x2,y2),(x3,y3)]
    three.sort(key=operator.itemgetter(1,0))
    three = np.array(three)
    relthree =three-three[0]

    for key,value in Goldenpositions.items():
        value1=value-value[1]
        if check_ary_contains_ary(relthree,value):
            positions=value+three[0]
            info=[key]
            for pos in positions:
                info.append(tuple(pos))
            allpossibility.append(info)
        elif check_ary_contains_ary(relthree,value1):
            positions = value1+three[0]
            info=[key]
            for pos in positions:
                info.append(tuple(pos))
            allpossibility.append(info)


    random.shuffle(allpossibility)
    return allpossibility




class Partial_samples(object):
    """
    create a frame sample in which no pieces should be used to tile the target
    """
    def __init__(self,rows,columns,prob_fill,filltype):
        """
        :param rows: number of rows
        :param colums: number of colums
        :param prob_fill: if possible to fill a square, the probability to fill
        """
        self.rows = rows
        self.columns = columns
        self.prob_fill = prob_fill
        self.T = np.zeros([rows, columns], int)
        self.S = np.zeros([rows, columns, 2], int)
        self.M = np.zeros([rows, columns, 2], int)
        self.filltype = filltype
        self.PiD = 1

        self.unused = []
        for x in range(0, columns):
            for y in range(0, rows):
                self.unused.append((x, y))
        random.shuffle(self.unused)

    def fill_square(self):

        if self.filltype == 'Null':
            while len(self.unused)>0:
                (x, y) = self.unused.pop()
                if not self.getthree(x,y,tile=False):
                    if np.random.rand()<self.prob_fill:
                        self.T[y][x]=1
        elif self.filltype == 'Three':
            while len(self.unused) > 0:
                (x, y) = self.unused.pop()
                if not self.getfour(x,y):
                    self.T[y][x]=1
                    self.getthree(x,y,tile=True)
            self.cleartarget()
        else:
            raise TypeError('unknown fill type')

    def cleartarget(self):
        for x in range(0,self.columns):
            for y in range(0,self.rows):
                if self.M[y][x][0]==0:
                    self.T[y][x]=0

    def withinrange(self,x,y):
        if x>=0 and x<self.columns and y>=0 and y<self.rows:
            return True
        else:
            return False

    def getthree(self,x,y,tile):
        """
        check if square (x,y) can be combined with surrounding available squares to form threecovered piece
        :param x:
        :param y:
        :param tile: if can form, whether to tile it
        :return: sign (True:can form, False: can not form)
        """

        def dist(x1,y1,x2,y2):
            d=abs(x1-x2)+abs(y1-y2)
            return d

        def threetile(allpossible):
            tiled=False
            while len(allpossible)>0:
                possible=allpossible.pop()
                sign = True
                for (x,y) in possible[1:]:
                    if (not self.withinrange(x,y)) or self.M[y][x][0]!=0:
                        sign=False
                if sign:
                    assert len(possible[1:])==4, 'did not tile 4 square'
                    assert possible[0]!=0, 'shape not be 0'
                    for np,(x,y) in enumerate(possible[1:]):
                        self.M[y][x]=(possible[0],self.PiD)
                        self.S[y][x]=(possible[0],np)
                    tiled=True
                    self.PiD += 1
                    break

            return tiled

        exist=[]
        sign=False

        for x_ in np.arange(max(0,x-3), min(x+4, self.columns),1):
            for y_ in np.arange(max(0,y-3), min(y+4, self.rows),1):
                if self.T[y_][x_]==1 and dist(x,y,x_,y_)<=3 and self.M[y_][x_][0]==0:
                    exist.append((x_,y_))

        random.shuffle(exist)
        for ((x1,y1),(x2,y2)) in itertools.combinations(exist,2):
            threedist=[dist(x1,y1,x2,y2),dist(x1,y1,x,y),dist(x,y,x2,y2)]
            sortdist=sorted(threedist)
            if tuple(sortdist) in [(1,1,2),(1,2,3),(2,2,2)]:
                sign=True
                if tile:
                    allpossible=threetofour(x,y,x1,y1,x2,y2)
                    if threetile(allpossible):
                        break
                    else:
                        continue
                else:
                    break

        return sign

    def getfour(self,x,y):

        def dist(x1, y1, x2, y2):
            d = abs(x1 - x2) + abs(y1 - y2)
            return d

        exist=[]
        sign=False

        for x_ in np.arange(max(0, x - 3), min(x + 4, self.columns), 1):
            for y_ in np.arange(max(0, y - 3), min(y + 4, self.rows), 1):
                if self.T[y_][x_]==1 and dist(x,y,x_,y_)<=3:
                    exist.append((x_,y_))

        for ((x1, y1), (x2, y2),(x3, y3)) in itertools.combinations(exist, 3):
            positions=[(x,y),(x1, y1), (x2, y2),(x3, y3)]
            positions.sort(key=operator.itemgetter(1,0))
            positions=np.array(positions)
            relpositions=positions-positions[0]
            for key,value in Goldenpositions.items():
                if np.all(relpositions==value):
                    sign=True
                    break
            if sign:
                break

        return sign





if __name__=='__main__':
    with open('10_10.txt', 'w') as f:
        for prob_blank in np.arange(0.0, 0.6, 0.1):
            for n in range(0, 100):
                sample = Create_sample(10, 10, prob_blank)
                sample.add_pieces()
                data = {
                    'T': sample.T.tolist(),
                    'M': sample.M.tolist(),
                    'S': sample.S.tolist()
                }
                line = json.dumps(data)
                line += '\n'
                f.write(line)
                # check(sample.T,sample.M, sample.S)

    with open('20_20.txt', 'w') as f:
        for prob_blank in np.arange(0.0, 0.6, 0.1):
            for n in range(0, 100):
                sample = Create_sample(20, 20, prob_blank)
                sample.add_pieces()
                data = {
                    'T': sample.T.tolist(),
                    'M': sample.M.tolist(),
                    'S': sample.S.tolist()
                }
                line = json.dumps(data)
                line += '\n'
                f.write(line)
                # check(sample.T,sample.M, sample.S)

    with open('50_50.txt', 'w') as f:
        for prob_blank in np.arange(0.0,0.6,0.1):
            for n in range(0, 100):
                sample=Create_sample(50,50,prob_blank)
                sample.add_pieces()
                data={
                    'T': sample.T.tolist(),
                    'M': sample.M.tolist(),
                    'S': sample.S.tolist()
                }
                line = json.dumps(data)
                line += '\n'
                f.write(line)
                #check(sample.T,sample.M, sample.S)
