""" MA3.py

Student: Nora Lundberg
Mail: nora.lundberg.3081@student.uu.se
Reviewed by:
Date reviewed:

"""
import random
import matplotlib.pyplot as plt
import math as m
import concurrent.futures as future
from statistics import mean 
from time import perf_counter as pc
from numba import njit

# Exc1
def approximate_pi(n):
    # n is the number of points
    x_in = []
    x_out = []
    y_in = []
    y_out = []
    for i in range(n):
        x = random.uniform(-1, 1)
        y = random.uniform(-1, 1)
        if (x**2 + y**2) <= 1:
            x_in.append(x)
            y_in.append(y)
        else: 
            x_out.append(x)
            y_out.append(y)

   #plt.scatter(x_in, y_in, color="red", s=1)
   #plt.scatter(x_out, y_out, color="blue", s=1)
   #plt.savefig(f"pi_{n}.png")
   #plt.close()
    pi_approx = 4 * (len(x_in)/n)

    print(n)
    print(pi_approx)
    return pi_approx

    

# Exc2, approximation              V_d är volymen av en d-dimensionell hypersfär V_d = (kubens volym) · n_c / n
def sphere_volume(n, d): 
    # n is the number of points

    # d is the number of dimensions of the sphere 
    #punkt = [random.uniform(-1,1) for i in range(d)]
    #kvadrater = list(map(lambda x: x**2 , punkt))
    #summa = sum(kvadrater)
    punkter = [[random.uniform(-1,1) for i in range(d)] for i in range(n)]
    inne = filter(lambda p:sum(map(lambda x: x**2, p)) <= 1, punkter)
    n_c = len(list(inne))
    
    return 2**d * n_c/n

 

#Exc2, real value
def hypersphere_exact(n, d):
    # n is the number of points

    # d is the number of dimensions of the sphere 

    V_dr = (m.pi**(d/2))/(m.gamma((d/2) + 1))
    return V_dr


#Exc3: numba version
@njit
def sphere_volume_numba(n:int, d:int)->float:
    # n is the number of points

    # d is the number of dimensions of the sphere
    #np is the number of processes
    n_c = 0.0
    for i in range(n):
        summa = 0
        for j in range(d):
            summa += (random.uniform(-1, 1))**2
            
        if summa <= 1:
            n_c += 1
    return 2**d * n_c / n
                


#Exc4: parallel code - parallelize actual computations by splitting data
def sphere_volume_parallel2(n, d, np=10):
    # n is the number of points
    # d is the number of dimensions of the sphere
    # np is the number of processes
    
    with future.ProcessPoolExecutor() as ex:
        proc_lst = []
        for i in range(np):
            proc_lst.append(ex.submit(sphere_volume, n//np, d))
        s = 0.0
        for proc in proc_lst:
            s += proc.result()
        return s/np

     

def main():
    # Exc1
    dots = [1000, 10000, 100000]
    for n in dots:
        approximate_pi(n)

    # Exc2
    n = 100000
    d = 2
    sphere_volume(n, d)
    print(f"Actual volume of {d} dimentional sphere = {hypersphere_exact(n,d)}")
    print(f"Approximation od {d} dimentional sphere = {sphere_volume(n,d)}")

    n = 100000
    d = 11
    sphere_volume(n, d)
    print(f"Actual volume of {d} dimentional sphere = {hypersphere_exact(n,d)}")
    print(f"Approximation od {d} dimentional sphere = {sphere_volume(n,d)}")

    # Exc3
    n = 1000000
    d = 11
    for n in range(1,4):
        start = pc()
        sphere_volume(n, d)
        stop = pc()
        print(f"Exc3: Sequential time of {d} and {n}: {stop-start}")
    print("What is numba time?")

    for n in range(1,4):
        start = pc()
        sphere_volume_numba(n, d)
        stop = pc()
        print(f"Exc3: Numba time of {d} and {n}: {stop-start}")

    # Exc4
    n = 1000000
    d = 11
    start = pc()
    sphere_volume(n, d)
    stop = pc()
    print(f"Exc4: Sequential time of {d} and {n}: {stop-start}")
    print("What is parallel time?")

    
    start = pc()
    sphere_volume_parallel2(n, d)
    stop = pc()
    print(f"Exc4: Parallel time of {d} and {n}: {stop-start}")

    
    

if __name__ == '__main__':
    main()