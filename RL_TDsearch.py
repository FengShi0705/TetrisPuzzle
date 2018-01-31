# this script use TD search:
# 1. use golden data to assist the search:
# (1). Use e-greedy instead of UCB, otherwise it will always go through the golden data
# (2). three steps: Select, Store and Train