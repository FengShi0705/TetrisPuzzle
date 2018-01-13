#----------------------
#     multi-armed bandit test
#----------------------
from numpy.random import rand
import numpy as np
from math import sqrt,log


def multiarmed_bandit(T,p1,p2,p3):
    A = {'W':0.0, 'Q': 0.0, 'n': 0, 'p':p1, 'id':'A'}
    B = {'W':0.0, 'Q': 0.0, 'n': 0, 'p':p2, 'id':'B'}
    C = {'W':0.0, 'Q': 0.0, 'n': 0, 'p':p3, 'id':'C'}
    arms=[A,B,C]
    hist=[]

    for t in range(1, T+1):
        values = []
        for arm in arms:
            v = arm['Q'] + ( sqrt(2*t) / (1+arm['n']) )
            values.append((v,arm))

        #if len(values)==len(arms):
        _,marm = max(values,key= lambda s:s[0])
        take(marm)
        hist.append(marm['id'])

    return arms,hist

def take(arm):
    print(arm['id'])
    r = rand(1)[0]
    if r<=arm['p']:
        arm['W'] += 1
        arm['n'] += 1
        arm['Q'] = arm['W']/arm['n']
    else:
        arm['n'] += 1
        arm['Q'] = arm['W'] / arm['n']

    return


