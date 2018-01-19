import multiprocessing

def f(a,b=1):
    return {'c':[a,b]}

if __name__=='__main__':
    pool = multiprocessing.Pool(4)
    results = pool.starmap(f,[('model',) for i in range(4)])
    print(results)
