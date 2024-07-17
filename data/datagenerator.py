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

    """
    Generates data for task 1

    Args:
        l = length of signal
        m = point selected at random at which the signal starts
        d[] = list of dimensions for sinosuidal waves
        sigs = list of signal-dimension for waves to be classified
        Cp = Coincedent signal point or number 
        Ap = Any signal point or number
        s1,2,3 = Signal Types
        p or Tar = Singal selected or Target Signal respectively 
    """

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

    """
    Generates priliminary data for task 1 signals and pointers

    Args:
        l = length of signal
        m = point selected at random at which the signal starts, 0 in this case
        d[] = list of dimensions for sinosuidal waves
        sigs = list of signal-dimension for waves to be classified
        Cp = Coincedent signal point or number
        Ap = Any signal point or number
        s1,2,3 = Signal Types
        p or Tar = Singal selected or Target Signal respectively 
    """

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



def task2(net, sequence_length, jumps):

    """
    Generates data for task 2

    Args:
        l = length of signal
        m = point selected at random at which the signal starts
        d1 and d2 = signal dimensions
        s1,2,3 = Target/pointer signals
        p = list of pointers except p[-1], which is the target 
    """

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

        l = random.randint(10, sequence_length // (jumps + 1) )

        m = [random.randint(0, ((sequence_length // (jumps + 1)) - l))]
        for i in range(jumps):
            m.append(random.randint(m[-1]+l, (m[-1] + (sequence_length // (jumps + 1)))))

        p = [random.randint(1,3) for _ in range(jumps+1)]

        for i in range(l):
            d1[m[0]] = (np.sin(i*2*np.pi/(l-1)))/2

            j = i/(l-1)
            for k in range(jumps):
                if p[k] == 1:
                    d1[m[k+1]] = s1(j)
                    d2[m[k]] = s1(j)
                if p[k] == 2:
                    d1[m[k+1]] = s2(j)
                    d2[m[k]] = s2(j)
                if p[k] == 3:
                    d1[m[k+1]] = s3(j)
                    d2[m[k]] = s3(j)
                
                m[k] += 1
            
            if p[jumps] == 1:
                d2[m[jumps]] = s1(j)
            if p[jumps] == 2:
                d2[m[jumps]] = s2(j)
            if p[jumps] == 3:
                d2[m[jumps]] = s3(j)
            
            m[jumps] += 1  

        Batch = (d1, d2, p[-1]-1)
        data.append(Batch)

    print('data stored')

    return data