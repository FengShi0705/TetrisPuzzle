import RL_totheEnd
from mysolution import Create_sample
import utils

sample = Create_sample(20, 20, 0.4)
sample.add_pieces()
target, solution = sample.T, sample.S

rightdata = RL_totheEnd.solve_game(target,solution)
utils.validation(target,rightdata)