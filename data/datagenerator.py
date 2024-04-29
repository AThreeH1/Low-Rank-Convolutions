data = []

for r in range(1000):

    # initialising dimentions d1 and d2 with random numbers
    d1 = []
    d2 = []

    for i in range(100):
        d1.append((random.random()) - 0.5)
        d2.append((random.random()) - 0.5)

    # initializing length of signal 'l' in dimension d1 & d2
    l = (random.randint(10, len(d1)/2))
    # print('length of main signal = ', l)

    # initializing the point 'm' from where the signal starts
    m = (random.randint(0, (len(d1)-l)))
    # print('The signal starts from index ', m)

    # # Initialising the three signals s1, s2 and s3 in dimension d2
    # print('Signal s1 is triangular wave')
    # print('Signal s2 is square wave')
    # print('Signal s3 is downward semicircular wave')

    # s1 - triangular wave
    def s1(t):
        if t < 0.5:
            return t
        else:
            return 1 - t

    # s2 - square wave
    def s2(t):
        if 0 <= t < 0.5:
            return 0.5
        else:
            return -0.5

    # s3 - lower semicircle
    def s3(t):
        return (-(0.5**2 - (t - 0.5)**2)**0.5)

    # selecting either of s1, s2 or s3 at random
    p = random.randint(1,3)
    # print('Target signal of s', p, ' is selected')

    # Inputting the signals in the respective dimensions
    for i in range(l):

        d1[m] = (np.sin(i*2*np.pi/(l-1)))/2

        j = i/(l-1)
        if p == 1:
            d2[m] = s1(j)
        if p == 2:
            d2[m] = s2(j)
        if p == 3:
            d2[m] = s3(j)

        m += 1

    if len(d2) - m >= 10:

        ps = random.randint(1,3)
        # print('Post signal of s', ps, ' is selected')

        Q = random.randint(10, min((len(d1) - m), 50))
        # print('Post noise is of length = ', Q)

        R = random.randint(0, (len(d1) - m - Q))

        post = m + R
        # print('Post noise starts from index ', post)

        for i in range(Q):
            j = i/(Q-1)
            if ps == 1:
                d2[post] = s1(j)
            if ps == 2:
                d2[post] = s2(j)
            if ps == 3:
                d2[post] = s3(j)

            post += 1

    if (m - l) >= 10:

        bs = random.randint(1,3)
        # print('Pre signal of s', bs, ' is selected')

        Qn = random.randint(10, min((m - l), 50))
        # print('Pre noise is of length = ', Qn)

        Rn = random.randint(0, (m - l - Qn))

        pre = Rn
        # print('Pre noise starts from index ', pre)

        for i in range(Qn):
            j = i/(Qn-1)
            if bs == 1:
                d2[pre] = s1(j)
            if bs == 2:
                d2[pre] = s2(j)
            if bs == 3:
                d2[pre] = s3(j)

            pre += 1

    Batch = (d1, d2, p-1)
    data.append(Batch)

print('data stored')