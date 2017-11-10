from mysolution import Partial_samples
import main
import utils
import mainissac
import numpy as np
from copy import deepcopy
from check import check

def test(target, solufunc):
    solution,S = solufunc(target,None,None)
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
        print('total error: {}'.format(total_error))

        return total_error


error=np.empty([8, 3, 3, 20], object)
algorithms = [main.Tetris_v1, main.Tetris_v2, main.Tetris_v3, main.Tetris_v4, main.Tetris_v5, main.Tetris_v6, main.Tetris_v7]


for s,size in enumerate([10,20,50]):
    for d,density in enumerate([0.6,0.7,0.8]):
        for n in range(0, 20):
            T = utils.generate_target(width=size, height=size, density=density)
            for i,alg in enumerate(algorithms):
                T_ = deepcopy(T)
                e = test(T_, alg)
                error[i][s][d][n]=e
            T_ = deepcopy(T)
            e = test(T_,mainissac.Tetris)
            error[7][s][d][n]=e
            print('---------finish size{} density{} n{}'.format(size,density,n))



np.save('real_test_results.npy',error)
print('all finished')


