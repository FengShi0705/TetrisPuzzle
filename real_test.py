from mysolution import Partial_samples
import main
import utils
from check import check

def test(target, solufunc):
    solution,S = solufunc(target)
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


e_main = 0
for n in range(0,100):
    #sample = Partial_samples(10, 10, 1, 'Three')
    #sample.fill_square()
    #T = sample.T.tolist()
    T = utils.generate_target(width=50, height=50, density=0.6)

    e_main += test(T, main.Tetris)


print('----------- For solvable samples --------------')
print('avg e_main: {}'.format(e_main/100))


