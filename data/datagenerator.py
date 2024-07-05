import os
import sys

# Get the directory of the current script
current_dir = os.path.dirname(__file__)

# Append the parent directory to the system path
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(parent_dir)

from utils.imports import *


import random
import numpy as np

def DataGenerate(net, sequence_length, dim):

# l = length of signal
# m = point selected at random at which the signal starts
# d[] = list of dimensions for sinosuidal waves
# sigs = list of signal-dimension for waves to be classified
# Cp = Coincedent signal point or number 
# Ap = Any signal point or number
# S1,2,3 = Signal Types
# p or Tar = Singal selected or Target Signal respectively 

    dim = dim - 1
    data = []

    for r in range(net):

        # initialising dimentions with random numbers
        d = [[] for i in range(dim)]
        sigs = []

        for i in range(sequence_length):
            for j in range(dim):
                d[j].append((random.random()) - 0.5)
            sigs.append((random.random()) - 0.5)

        def s1(t):
            if t < 0.5:
                return t
            else:
                return 1 - t
        def s2(t):
            if 0 <= t < 0.5:
                return 0.5
            else:
                return -0.5
        def s3(t):
            return (-(0.5**2 - (t - 0.5)**2)**0.5)

        Cp = random.randint(1, dim)
        Ap = [i+1 for i in range(dim+1) if (i+1) != Cp]
        l = random.randint(10, sequence_length // (dim + 1))
        m = random.randint(0, (sequence_length // (dim + 1)) - l)

        for sn in range(dim+1):

            p = random.randint(1,3)

            if (sn+1) == Cp:
                Tar = p
            for i in range(l):
                for k in range(dim):
                    if (sn+1) != Ap[k]:                    
                        d[k][m] = (np.sin(i*2*np.pi/(l-1)))/2

                j = i/(l-1)
                if p == 1:
                    sigs[m] = s1(j)
                if p == 2:
                    sigs[m] = s2(j)
                if p == 3:
                    sigs[m] = s3(j)

                m += 1

            l = (random.randint(10, sequence_length//(dim+1)))
            m = random.randint(m, ((sn+2)*(sequence_length//(dim+1))-l))

        Batch = (*d[:dim], sigs, Tar-1)
        data.append(Batch)

    print('data stored')
    
    return data


def SimpleDataGenerate(net, sequence_length):

# l = length of signal
# m = point selected at random at which the signal starts
# d[] = list of dimensions for sinosuidal waves
# sigs = list of signal-dimension for waves to be classified
# Cp = Coincedent signal point or number 
# Ap = Any signal point or number
# S1,2,3 = Signal Types
# p or Tar = Singal selected or Target Signal respectively 

    dim = 1
    data = []

    for _ in range(net):

        # initialising dimentions with random numbers
        d = [[] for i in range(dim)]
        sigs = []

        for i in range(sequence_length):
            for j in range(dim):
                d[j].append((random.random()) - 0.5)
            sigs.append((random.random()) - 0.5)

        def s1(t):
            if t < 0.5:
                return t
            else:
                return 1 - t
        def s2(t):
            if 0 <= t < 0.5:
                return 0.5
            else:
                return -0.5
        def s3(t):
            return (-(0.5**2 - (t - 0.5)**2)**0.5)

        Cp = random.randint(1, dim)
        Ap = [i+1 for i in range(dim+1) if (i+1) != Cp]
        l = sequence_length
        m = 0

        p = random.randint(1,3)

        Tar = p
        

        for i in range(l):
                
            d[0][m] = (np.sin(i*2*np.pi/(l-1)))/2

            j = i/(l-1)
            if p == 1:
                sigs[m] = s1(j)
            if p == 2:
                sigs[m] = s2(j)
            if p == 3:
                sigs[m] = s3(j)

            m += 1


        Batch = (*d[:dim], sigs, Tar-1)
        data.append(Batch)

    print('data stored')
    
    return data



def task2(net, sequence_length):

# l = length of signal
# m = point selected at random at which the signal starts
# d[] = list of dimensions for sinosuidal waves
# sigs = list of signal-dimension for waves to be classified
# Cp = Coincedent signal point or number 
# Ap = Any signal point or number
# S1,2,3 = Signal Types
# p or Tar = Singal selected or Target Signal respectively 

    data = []

    for r in range(net):

        # initialising dimentions with random numbers
        d1 = []
        d2 = []

        for i in range(sequence_length):
            d1.append((random.random()) - 0.5)
            d2.append((random.random()) - 0.5)

        def s1(t):
            if t < 0.5:
                return t
            else:
                return 1 - t
        def s2(t):
            if 0 <= t < 0.5:
                return 0.5
            else:
                return -0.5
        def s3(t):
            return (-(0.5**2 - (t - 0.5)**2)**0.5)

        l = random.randint(10, sequence_length // 2)

        m1 = random.randint(0, (sequence_length // 2) - l)
        m2 = random.randint(m1+l, ((sequence_length // 2) - l) + m1+l)

        p1 = random.randint(1,3)
        p2 = random.randint(1,3)

        for i in range(l):
            d1[m1] = (np.sin(i*2*np.pi/(l-1)))/2

            j = i/(l-1)
            if p1 == 1:
                d1[m2] = s1(j)
                d2[m1] = s1(j)
            if p1 == 2:
                d1[m2] = s2(j)
                d2[m1] = s2(j)
            if p1 == 3:
                d1[m2] = s3(j)
                d2[m1] = s3(j)

            if p2 == 1:
                d2[m2] = s1(j)
            if p2 == 2:
                d2[m2] = s2(j)
            if p2 == 3:
                d2[m2] = s3(j)

            m1 += 1
            m2 += 1

        Batch = (d1, d2, p2-1)
        data.append(Batch)

    print('data stored')
    
    return data