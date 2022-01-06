import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt(open("D:\hello\dun.txt","r"), delimiter=",")
x1=data[:, 0]
y=data[:, 1]

# Building the model
t1 =0
t2 =0
print(x1.shape)
m = len(y)
alpha = 0.01  # The learning Rate
ht = 0
n = float(len(x1)) # Nut1ber of elet1ents in X

l=[]
beta = 0.9
v1 = 0
v2 = 0
s1 = 0
s2 = 0

# Performing Gradient Descent
for i in range(1,20):
    for j in range(10):
        ht = t1*x1[j:10+j] + t2
        D_t1 = (-2/n) * sum(x1[j:10+j] * (y[j:10+j] - ht) )  # Derivative wrt t1
        D_t2 = (-2/n) * sum( (y[j:10+j] - ht) )

        v1 = beta*v1 + (1-beta)*D_t1
        v2 = beta*v2 + (1-beta)*D_t2

        s1 = beta*s1 + (1-beta)*(D_t1**2)
        s2 = beta*s2 + (1-beta)*(D_t2**2)

        t1 = t1 - alpha * (v1 / np.sqrt(s1 + 1e-8))  # Update t1
        t2 = t2 - alpha * (v2 / np.sqrt(s2 + 1e-8))  # Update t2
 
        k = (1.0/(2*n)) * sum((ht - y[j:10+j])**2)
        l.append(k)

print(len(l))
plt.plot(range(1,191),l)
plt.show()




