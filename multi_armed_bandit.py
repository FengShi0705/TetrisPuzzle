#----------------------
#     multi-armed bandit test
#----------------------
from numpy.random import rand
import numpy as np
from math import sqrt,log


def multiarmed_bandit(c,T,p1,p2,q1,q2):
    A = {'W':0.0, 'Q': 0.0, 'n': 0, 'p':p1, 'q':q1, 'id':'A'}
    B = {'W':0.0, 'Q': 0.0, 'n': 0, 'p':p2, 'q':q2, 'id':'B'}
    #C = {'W':0.0, 'Q': 0.0, 'n': 0, 'p':p3, 'q':q3, 'id':'C'}
    arms=[A,B]
    hist=[]

    for t in range(1, T+1):
        values = []
        for arm in arms:
            v = arm['Q'] + (0.5*sqrt(2*t) / (1+arm['n']) )
            values.append((v,arm))

        #if len(values)==len(arms):
        _,marm = max(values,key= lambda s:s[0])
        take(marm)
        hist.append(marm['id'])

    return arms,hist

def take(arm):
    print(arm['id'])
    arm['W'] += arm['q']
    arm['n'] += 1
    arm['Q'] = arm['W']/arm['n']
    #r = rand(1)[0]
    #if r<=arm['q']:
    #    arm['W'] += 1
    #    arm['n'] += 1
    #    arm['Q'] = arm['W']/arm['n']
    #else:
    #    arm['W'] -= 1
    #    arm['n'] += 1
    #    arm['Q'] = arm['W'] / arm['n']

    return


