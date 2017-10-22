import numpy as np
import math
import matplotlib.pyplot as plt

def load_file(filename = "./hw1_8_train.dat"):
    data = []
    for line in open(filename, 'r'):
        item = line.rstrip()
        str_x, str_y = item.split("\t")
        x = str_x.split(" ")
        x = [float(i) for i in x]
        x.append(1)
        x.append(int(str_y))
        data.append(x)
    return (data)

def sign(tmp):
    if (tmp > 0):
        return (1.0)
    else:
        return (-1.0)

def main():
    # load data
    data = np.array(load_file())
                
    update_count_stat = []
    m=2000
    data_size=400
    #2000 times
    for seed in range(m):
        np.random.seed(seed)
        np.random.shuffle(data)
        w = np.zeros(5)

        update_count = 0
        i = 0
        y = data[:,5]

        while(True):
            idx = i % data_size
            y_tmp = np.dot(data[:,:5], np.transpose(w)) 
            y_hat = np.array(list(map(sign, y_tmp)))
            err = sum(np.absolute(y - y_hat))
            if(err > 0):
                if (y[idx] != y_hat[idx]):
                    w += data[idx, :5]*y[idx]
                    update_count += 1
                i += 1
            else:
                update_count_stat.append(update_count)
                break
    print("avg update time:", sum(update_count_stat)/len(update_count_stat))
    plt.hist(update_count_stat, bins=15)
    plt.show()

if __name__ == "__main__":
    main()

