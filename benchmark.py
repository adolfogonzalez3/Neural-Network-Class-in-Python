
import operator
from time import time
from multiprocessing import Process

import numpy as np
from pynn import Network, Matrix2d
from tqdm import trange

def run(matrix_A, matrix_B, N):
    for i in range(N):
        matrix_A @ matrix_B

def main():
    trials = int(1e3)
    test_A = Matrix2d.random(100, 100)
    test_B = Matrix2d.random(100, 100)
    N = 100*100
    test_C = [i % 100 for i in range(100*100)]

    
    
    #begin = time()
    #for _ in trange(trials):
    #    test_A / test_B
    #print(f'Time elapsed: {time() - begin}')

    num_procs = 8
    begin = time()
    procs = [
        Process(target=run, args=(test_A, test_B, trials//num_procs))
        for _ in range(num_procs)
    ]
    for proc in procs:
        proc.start()
    for proc in procs:
        proc.join()
    print(f'Time elapsed: {time() - begin}')



if __name__ == '__main__':
    main()
