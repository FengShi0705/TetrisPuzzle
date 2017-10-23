# ####################################################
# DE2-COM2 Computing 2
# Individual project
#
# Title: PERFORMANCE TEST
# Author: Liuqing Chen, Feng Shi, Isaac Engel (13th September 2017)
# Last updated: 13th September 2017
# ####################################################

from main import Tetris_v1,Tetris_v2,Tetris_v3,Tetris_v4,Tetris_v5,Tetris_v6,Tetris_v7
import utils
import timeit
import time
import mainissac
import matplotlib.pyplot as plt
from copy import deepcopy
import threading

class myAlg:
    def __init__(self,func,target,fig,ax):
        self.func=func
        self.target = target
        self.fig = fig
        self.ax = ax

    def run(self):
        begin_time = time.time()
        self.solution, self.S = self.func(self.target,self.fig, self.ax)
        end_time = time.time()
        self.time = end_time - begin_time
        return

class issacAlg:
    def __init__(self, func, target, fig, ax):
        self.func = func
        self.target = target
        self.fig = fig
        self.ax = ax

    def run(self):
        begin_time = time.time()
        self.solution = self.func(self.target, self.fig, self.ax)
        end_time = time.time()
        self.time = end_time - begin_time
        return


        # Example target shape
#target = [[1, 0, 0, 0], [1, 1, 1, 1], [1, 0, 1, 1], [1, 1, 1, 0]]  # NOTE: in your test, you may not use this example.
#target = [[1,0],[1,1]]
#target = [[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,1,1,0,0,0],[0,0,0,1,1,0,0,0,0],[0,0,0,1,1,0,0,0,0],[0,0,0,1,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0]]
# Uncomment the following line to generate a random target shape
target = utils.generate_target(width=20, height=20, density=0.6)  # NOTE: it is recommended to keep density below 0.8
target_ = deepcopy(target)

# show target
plt.ion()
fig = plt.figure()
ax1, ax2 = utils.showtarget(target, fig)

plt.pause(1)
time.sleep(1)
my = myAlg(Tetris_v7,target,fig,ax1)
issac = issacAlg(mainissac.Tetris,target_,fig,ax2)
my.run()
utils.Mark_Wrong_square(target,my.solution,ax1)
issac.run()
utils.Mark_Wrong_square(target_,issac.solution,ax2)

# finalize visualisation
plt.ioff()
plt.show()

valid, missing, excess, error_pieces = utils.check_solution(target, my.solution)  # checks if the solution is valid

if not valid:

    print("The solution is not valid!")

else:  # if the solution is valid, test time performance and accuracy

    # TIME PERFORMANCE
    # There will be three different values of the parameter 'target' with increasing complexity in real test.

    time_set = my.time

    if time_set > 600:

        print("Time is over 10 minutes! The solution is not valid")

    else:

        print("Time performance")
        print("----------------")
        print("The running time was {:.5f} seconds.\n".format(time_set))

        # ACCURACY

        print("Accuracy")
        print("--------")

        if len(error_pieces) == 0:
            print('All pieces are labelled with correct shapeID and pieceID.')
        else:
            print('WARNING: {} pieces have a wrong shapeID. They are labelled in image of the solution, and they are: {}.'
                  .format(len(error_pieces), error_pieces))

        total_blocks = sum([sum(row) for row in target])
        total_blocks_solution = total_blocks - missing + excess

        print("The number of blocks in the TARGET is {:.0f}.".format(total_blocks))
        print("The number of blocks in the SOLUTION is {:.0f}.".format(total_blocks_solution))
        print("There are {} MISSING blocks ({:.4f}%) and {} EXCESS blocks ({:.4f}%).\n".format
              (missing, 100 * missing / total_blocks, excess, 100 * excess / total_blocks))

        # VISUALISATION
        # NOTE: for large sizes (e.g., 100x100), visualisation will take several seconds and might not be that helpful.
        # Feel free to comment out the following lines if you don't need the visual feedback.

        print("Displaying solution...")
        utils.visualisation(target, my.solution)



#----------------


valid, missing, excess, error_pieces = utils.check_solution(target, issac.solution)  # checks if the solution is valid

if not valid:

    print("The solution is not valid!")

else:  # if the solution is valid, test time performance and accuracy

    # TIME PERFORMANCE
    # There will be three different values of the parameter 'target' with increasing complexity in real test.

    time_set = issac.time

    if time_set > 600:

        print("Time is over 10 minutes! The solution is not valid")

    else:

        print("Time performance")
        print("----------------")
        print("The running time was {:.5f} seconds.\n".format(time_set))

        # ACCURACY

        print("Accuracy")
        print("--------")

        if len(error_pieces) == 0:
            print('All pieces are labelled with correct shapeID and pieceID.')
        else:
            print('WARNING: {} pieces have a wrong shapeID. They are labelled in image of the solution, and they are: {}.'
                  .format(len(error_pieces), error_pieces))

        total_blocks = sum([sum(row) for row in target])
        total_blocks_solution = total_blocks - missing + excess

        print("The number of blocks in the TARGET is {:.0f}.".format(total_blocks))
        print("The number of blocks in the SOLUTION is {:.0f}.".format(total_blocks_solution))
        print("There are {} MISSING blocks ({:.4f}%) and {} EXCESS blocks ({:.4f}%).\n".format
              (missing, 100 * missing / total_blocks, excess, 100 * excess / total_blocks))

        # VISUALISATION
        # NOTE: for large sizes (e.g., 100x100), visualisation will take several seconds and might not be that helpful.
        # Feel free to comment out the following lines if you don't need the visual feedback.

        print("Displaying solution...")
        utils.visualisation(target, issac.solution)


