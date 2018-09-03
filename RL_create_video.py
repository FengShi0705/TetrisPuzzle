from mysolution import Create_sample
from utils import validation
import RL_totheEnd
import RL_naive_score
import RL_naive_V



sample = Create_sample(20, 20, 0.4)
sample.add_pieces()
target, solution = sample.T, sample.S
rightdata = RL_totheEnd.solve_game(target, solution)
data_naive_v, _, score_naive_v = RL_naive_V.Game(target,400, None,None).play()

validation(target,rightdata)

