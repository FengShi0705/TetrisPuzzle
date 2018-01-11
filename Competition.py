import main
import utils
import numpy as np
from copy import deepcopy
import mainissac
import time

def test(target, solufunc):
    starttime=time.time()
    solution,S = solufunc(target,None,None)
    endtime=time.time()
    valid, missing, excess, error_pieces = utils.check_solution(target, solution)

    if not valid:

        raise TypeError("The solution is not valid!")

    else:  # if the solution is valid, test time performance and accuracy

        # TIME PERFORMANCE
        # There will be three different values of the parameter 'target' with increasing complexity in real test.


        # ACCURACY


        if len(error_pieces) != 0:
            raise TypeError('Wrong shape')

        total_blocks = sum([sum(row) for row in target])

        total_error = (100 * missing / total_blocks) + (100 * excess / total_blocks)
        #print('total error: {}'.format(total_error))

        return total_error,endtime-starttime

filename_list = ['targets/20x20', 'targets/100x100', 'targets/250x250']
algorithms = [main.Tetris_v7, main.Tetris_v3]

for f,filename in enumerate(filename_list):
    for i,alg in enumerate(algorithms):
        file = open(filename)
        target = eval(file.read())
        file.close()
        error,T = test(target,alg)
        accur=100.0-error
        print('algorithm {} target {}: Time {} Accuracy {}'.format(i,f,T,accur))

"""
---------------------results:--------------------
"D:\Program Files\Python35_x64\python.exe" D:/CodeGit/TetrisPuzzle/Competition.py
2017-12-06 17:28:27.380550: W C:\tf_jenkins\home\workspace\rel-win\M\windows\PY\35\tensorflow\core\platform\cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use AVX instructions, but these are available on your machine and could speed up CPU computations.
2017-12-06 17:28:27.381550: W C:\tf_jenkins\home\workspace\rel-win\M\windows\PY\35\tensorflow\core\platform\cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use AVX2 instructions, but these are available on your machine and could speed up CPU computations.
Model restored.
algorithm 0 target 0: Time 11.240000009536743 Accuracy 99.0
Model restored.
algorithm 1 target 0: Time 29.343000173568726 Accuracy 100.0
Model restored.
algorithm 2 target 0: Time 63.785199880599976 Accuracy 100.0
Model restored.
algorithm 3 target 0: Time 29.430000066757202 Accuracy 100.0
Model restored.
algorithm 4 target 0: Time 29.20860004425049 Accuracy 99.0
Model restored.
algorithm 5 target 0: Time 11.345999956130981 Accuracy 98.0
Model restored.
algorithm 6 target 0: Time 11.09000015258789 Accuracy 100.0
algorithm 7 target 0: Time 0.00599980354309082 Accuracy 97.0
Model restored.
algorithm 0 target 1: Time 411.1884000301361 Accuracy 99.22625400213447
Model restored.
algorithm 1 target 1: Time 1052.7765998840332 Accuracy 99.43970117395945
Model restored.
algorithm 2 target 1: Time 2678.0411999225616 Accuracy 99.67982924226254
Model restored.
algorithm 3 target 1: Time 1056.6531999111176 Accuracy 99.51974386339381
Model restored.
algorithm 4 target 1: Time 1052.1791999340057 Accuracy 99.43970117395945
Model restored.
algorithm 5 target 1: Time 405.7409999370575 Accuracy 99.59978655282818
Model restored.
algorithm 6 target 1: Time 401.4865999221802 Accuracy 99.57310565635005
algorithm 7 target 1: Time 0.4790000915527344 Accuracy 98.93276414087514
Model restored.
algorithm 0 target 2: Time 2426.210000038147 Accuracy 98.77333333333334
Model restored.
algorithm 1 target 2: Time 6241.291599988937 Accuracy 98.208
Model restored.
algorithm 2 target 2: Time 16042.5644967556 Accuracy 98.992
Model restored.
algorithm 3 target 2: Time 6186.4113676548 Accuracy 98.80533333333334
Model restored.
algorithm 4 target 2: Time 6194.896018743515 Accuracy 98.86933333333333
Model restored.
algorithm 5 target 2: Time 2448.7084448337555 Accuracy 98.752
Model restored.
algorithm 6 target 2: Time 2417.2880988121033 Accuracy 98.86933333333333
algorithm 7 target 2: Time 1.4141414165496826 Accuracy 97.57866666666666

Process finished with exit code 0

"""