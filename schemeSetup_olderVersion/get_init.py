import numpy as np

if __name__ == "__main__":
    with open('evaluations/pred_mae.dat', 'r' ) as f:
        lines = f.readlines()
        f.closed
    n = np.array([ line.strip('\n').split()[0] for line in lines ] ,dtype =int )
    y = np.array([ line.strip('\n').split()[1] for line in lines ] ,dtype =float )
    n = n[-10:]
    y = y[-10:]
    val = n[y.argmin()] - 1
    print(val)
