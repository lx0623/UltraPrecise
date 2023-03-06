import numpy as np
np.random.seed(0)
array = np.random.normal(0.01, 0.0001, (100000000,1))
i = 0
t = 0
while(i<10000000):
    while( array[t]<0.01-0.0001 or array[t]>0.01+0.0001 ):
        t = t + 1
        if(t>100000000):
            print("ERROR!")
            os._exit
    # print(array[t])
    print("%.8f" % array[t])
    t=t+1
    i=i+1