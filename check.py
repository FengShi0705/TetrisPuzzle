#from somestudent import solution #import some student solutions
import numpy as np
import operator
import time
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image, ImageDraw

# the relative position of the four nodes wrt the first node ( most top(priority) - most left node )
goldenpos4 = {
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
    19: np.array([[0,0], [-1, 1], [0, 1], [-1, 2]])
}

def check(target,M,S):

    # Boundary check
    def boundary_check(target, M):
        missing = 0
        excess = 0

        nrows = len(target)
        ncols = len(target[0])

        if len(M) != nrows:
            print("Error: the matrices are not the same size!")
            return None

        for r in range(0, nrows):

            if len(target[r]) != ncols or len(M[r]) != ncols:
                print("Error: the matrices are not the same size!")
                return None

            for c in range(0, ncols):

                if target[r][c] == 0:
                    if tuple(M[r][c]) != (0, 0):
                        excess += 1
                elif target[r][c] == 1:
                    if tuple(M[r][c]) == (0, 0):
                        missing += 1
                else:
                    print("Error! The target matrix should only contain 1s and 0s")
                    return None

        return (missing, excess)

    # Shape check
    def checkposition(positions,shapeid):
        """
        check if positions of a piece corresponds with a speicific shape
        :param positions: poistions of blocks of a piece
        :param shapeid: the specified shape for this piece
        :return: whether or not the positions are correct
        """
        # the relative position of the last three node to the first node ( most top(priority) - most left node)
        goldenpositions = {
            1: np.array([[1, 0], [0, 1], [1, 1]]),
            2: np.array([[0, 1], [0, 2], [0, 3]]),
            3: np.array([[1, 0], [2, 0], [3, 0]]),
            4: np.array([[0, 1], [0, 2], [1, 2]]),
            5: np.array([[-2, 1], [-1, 1], [0, 1]]),
            6: np.array([[1, 0], [1, 1], [1, 2]]),
            7: np.array([[1, 0], [2, 0], [0, 1]]),
            8: np.array([[0, 1], [-1, 2], [0, 2]]),
            9: np.array([[1, 0], [2, 0], [2, 1]]),
            10: np.array([[1, 0], [0, 1], [0, 2]]),
            11: np.array([[0, 1], [1, 1], [2, 1]]),
            12: np.array([[0, 1], [1, 1], [0, 2]]),
            13: np.array([[-1, 1], [0, 1], [1, 1]]),
            14: np.array([[-1, 1], [0, 1], [0, 2]]),
            15: np.array([[1, 0], [2, 0], [1, 1]]),
            16: np.array([[1, 0], [-1, 1], [0, 1]]),
            17: np.array([[0, 1], [1, 1], [1, 2]]),
            18: np.array([[1, 0], [1, 1], [2, 1]]),
            19: np.array([[-1, 1], [0, 1], [-1, 2]])
        }

        matchM= ( np.array(positions[1:])-np.array(positions[0]) == goldenpositions[shapeid] )
        return np.all(matchM)

    def checkshape(M):
        """
        check if the pieces have the correct shape
        :param M: matrix containing the information of pieces, (shapeid, pieceid)
        :return:  id of pieces whose positions don't correspond with its shape
        """
        error_pieces = []
        Pieces={} #dictornay of pieces

        #extract all pieces from Matrix, and save their shapes and positions into Pieces
        for y,row in enumerate(M):
            for x,point in enumerate(row):
                shapeid=point[0]
                pieceid=point[1]
                if 0 in [pieceid,shapeid]:
                    assert pieceid == 0, "find shapeid is 0, but pieceid is {}".format(pieceid)
                    assert shapeid == 0, "find pieceid is 0, but shapeid is {}".format(shapeid)
                    continue
                elif pieceid in Pieces:
                    assert Pieces[pieceid]['shape']==shapeid, 'Error: difference shape for one piece'
                    Pieces[pieceid]['node'].append((x,y))
                else:
                    Pieces[pieceid]={}
                    Pieces[pieceid]['shape'] = shapeid
                    Pieces[pieceid]['node']=[(x,y)]

        # for each peice sort poisitions (left-right,up-down(priority)), and check if the position is correct
        for pid,piece in Pieces.items():
            piece['node'].sort(key=operator.itemgetter(1,0))
            assert len(piece['node'])==4, 'piece {} has {} blocks'.format(pid,len(piece['node']))
            if checkposition(piece['node'],piece['shape']):
                continue
            else:
                error_pieces.append(pid)

        #print(error_pieces)
        return error_pieces

    def checkshape_Matrixsolution(S):
        def popone(itemQ):
            [x,y,shape,pos]=itemQ.pop()
            for p in [0,1,2,3]:
                if p!=pos:
                    [x_,y_]= np.array([x,y])-( goldenpos4[shape][pos]-goldenpos4[shape][p] )
                    try:
                        itemQ.remove([x_,y_,shape,p])
                    except:
                        raise TypeError('Matrix S has wrong shape')


        allitems=[]
        for y,row in enumerate(S):
            for x,square in enumerate(row):
                if (square[0] != 0):
                    allitems.append([x,y,square[0],square[1]])
        while len(allitems)>0:
            assert (len(allitems)%4)==0,'Matrix S have wrong shape'
            popone(allitems)

        print('In matrix S, you get all pieces with the right shape ')

        return



    # Draw
    def visualisation(target,M,S):
        wrong_list = checkshape(M)
        Ty_len = len(target)
        Tx_len = len(target[0])
        Sy_len = len(M)
        Sx_len = len(M[0])

        fig,(ax1,ax2,ax3) = plt.subplots(1,3)  # Create figure and axes
        im = Image.new('RGB', (Tx_len+1,Ty_len+1), (255,255,255))  # white background-image
        dr = ImageDraw.Draw(im)
        ax1.imshow(im)  # Display the background-image
        ax2.imshow(im)
        ax3.imshow(im)

        #-------------------- Task Display ----------------------
        for y in range(Ty_len):
            row = target[y]
            for x in range(Tx_len):
                if row[x]==1:
                    ax1.add_patch(patches.Rectangle((x,y),0.88,0.88,color='b'))  # draw a block
        ax1.set_title('The Display of Task')

        # --------------- M Display ----------------------
        def get_color(num): # generate a random color
            np.random.seed(num)
            c=list(np.random.rand(3))
            c.append(1.0)
            return tuple(c)

        wrong_label_count = {}
        for y in range(Sy_len):
            row = M[y]
            for x in range(Sx_len):
                shape, num = row[x]
                if shape!=0:
                    ax2.add_patch(patches.Rectangle((x,y),0.88,0.88,color=get_color(num))) # draw a block
                    if num in wrong_list:
                        if wrong_label_count.setdefault(num,0)==0:
                            ax2.text(x,y+0.8,'{}'.format(num))  # add label to blocks that have wrong shapes
                            wrong_label_count[num]+=1
        ax2.set_title('The Display of M')

        # S display
        allsquares=[]
        for y,row in enumerate(S):
            for x,square in enumerate(row):
                if (square[0] !=0):
                    allsquares.append([x,y,square[0],square[1]])

        while len(allsquares)>0:
            [x,y,shape,pos]=allsquares.pop()
            color= list(np.random.rand(3))
            color.append(1.0)
            color=tuple(color)
            ax3.add_patch(patches.Rectangle((x, y), 0.88, 0.88, color=color))
            for p in [0,1,2,3]:
                if p != pos:
                    (x_,y_) = goldenpos4[shape][p]-goldenpos4[shape][pos]+np.array([x,y])
                    ax3.add_patch(patches.Rectangle((x_, y_), 0.88, 0.88, color=color))
                    allsquares.remove([x_,y_,shape,p])

        ax3.set_title('The Display of S')



        plt.show()


    #report time
    start_time = time.time()
    #M = solution(target)
    end_time = time.time()
    print ('You use {} second'.format(end_time-start_time))

    #report boundary_check
    results = boundary_check(target, M)
    if (results is not None):
        print ("There are", results[0], "missing blocks and", results[1], "excess blocks in matrix M.")
    results = boundary_check(target, S)
    if (results is not None):
        print("There are", results[0], "missing blocks and", results[1], "excess blocks in matrix S.")

    #report shape check
    error_pieces=checkshape(M)
    if len(error_pieces)==0:
        print ('In matrix M, you get all pieces with the right shape ')
    else:
        print ('In matrix M, you get {} pieces wrong (labelled in the image of solution ), which are {}.'.format(len(error_pieces),error_pieces))

    checkshape_Matrixsolution(S)

    #draw the figure
    visualisation(target,M,S)


