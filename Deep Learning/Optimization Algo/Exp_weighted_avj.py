import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

file = pd.read_csv('weatherHistory.csv')

time = file.iloc[:,0].values # date dataset
temp = file['Temperature (C)'].values # temperature per date

''' Collect the dataset '''
dataset = []
store = 1
check = 1
for i in range(len(time)):
    latest = int(time[i].split()[0][-2:])
    if latest == check:
        continue
    elif store == 365:
        break
    else:
        check = latest
        dataset.append([store, temp[i]])
        store += 1

''' Prediction '''
dataset = np.array(dataset)
pred = []
beta = 0.98
V_o = 0
for i in range(1, 1+len(dataset)):
    V_o = (beta*V_o) + ((1-beta)*int(dataset[i-1][1]))
    pred.append(V_o)


plt.scatter(dataset[:,0], dataset[:,1], color='b', )
plt.plot(dataset[:,0], pred, color='r')

plt.show()
