import numpy as np
import multiprocessing
from random import randint
import time 

def by_two(a):
    print('Multiplying {} by 2'.format(a))
    return a * 2

def add_numbers(a, b, c):
    print('Adding {}, {} and {}'.format(a, b, c))
    return a + b + c

def wait(a):
    wait_time = randint(0, 5)
    time.sleep(wait_time)
    print('Paused for {} seconds.'.format(wait_time))
    return a

def vector_out(a):
    return np.random.randn(4, )

workers = multiprocessing.cpu_count() - 1
print('Using %i workers.' % workers)

vector = [1, 2, 3, 4, 5]

start = time.time()
with multiprocessing.Pool(processes=workers) as pool:
        results = pool.map(vector_out, vector)
stop = time.time()
print(stop - start)

start = time.time()
results= []
for i in vector: 
    results.append(vector_out(i))
stop = time.time()
print(stop - start)

results = np.array(results).T
print(results)