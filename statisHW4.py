import math

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy.stats import norm
from scipy.stats import poisson
from scipy.stats import pearsonr
import pandas as pd


# Q1

# a
path = "C:/Users/Erezd/OneDrive/Desktop/heights.csv"
df = pd.read_csv(path)

x = df['HEIGHT']  # heightColumn
y = df['WEIGHT']  # weightColumn

plt.scatter(x , y)
plt.ylabel("Weight")
plt.xlabel("Height")

# add least squares line :
A = np.vstack([x, np.ones(len(x))]).T
m, c = np.linalg.lstsq(A, y, rcond=None)[0]
plt.plot(x, y, 'o', label='Ampirical data', markersize=10)
plt.plot(x, m*x + c, 'b', label='least squares line')
plt.legend()

# add resistant line :
X_sorted = np.sort(x)
y_values_first_third = []
y_values_last_third = []

third_q_item = X_sorted.item( math.ceil(len(x) * 1/3) -1)          # X 1/3 item
second_third_q_item = X_sorted.item( math.ceil(len(x) * 2/3) -1)   # X 2/3 item

# division of y values according to first and last third
for i in df.values:
    if i[1] < third_q_item:
        y_values_first_third.append(i[0])
    elif i[1] >= second_third_q_item:
        y_values_last_third.append(i[0])

# median X in each third
x_L =  X_sorted.item( math.ceil(len(x) * 1/6) -1)
x_H =  X_sorted.item( math.ceil(len(x) * 5/6) -1)

print("median left third x : ",x_L)
print("median right third x : ",x_H)

sorted_Y_left = sorted(y_values_first_third)
sorted_Y_right = sorted(y_values_last_third)

#print(sorted(y_values_first_third))
#print(sorted(y_values_last_third))

# median Y in each third
y_L =  sorted_Y_left[ math.ceil(len( sorted_Y_left ) * 0.5) - 1 ]
y_H = sorted_Y_right[ math.ceil(len( sorted_Y_right ) * 0.5) - 1]

print("median left third y : ",y_L)
print("median right third y : ",y_H)

b_RL = (y_H - y_L)/(x_H - x_L)
r = []
for i in df.values:
    r_i = i[0] - b_RL * i[1]
    r.append(r_i)

a_RL = sorted(r)[math.ceil(len(r) * 0.5) - 1]

plt.axline((0,a_RL) , slope = b_RL , color = 'r', label='resistant line')
plt.legend()

plt.show()

# d

x_sd = np.std(x)
y_sd = np.std(y)

r = pearsonr(x,y)[0]
b_LS = r * (y_sd / x_sd)
R_squared = r ** 2


print ("slope LS = ", b_LS)
print ("cor r = ",r)
print ("R^2 = ", R_squared)

# e - change path to file called height_without_one_point


# Q2

# a
X = norm.rvs(loc = 5, scale = 1, size = 30)
print (X)

# b
Y = 5 * X + 2
print(Y)

# c
r_XY = pearsonr(X,Y)[0]
print("r = ",r_XY)  # r = 1

# d
X_sd = np.std(X)
Y_sd = np.std(Y)

b_LS = r_XY * (Y_sd / X_sd)   # b = 5
print ("slope LS = ", b_LS)

#R_squared = r_XY ** 2

#print ("cor r = ",r_XY)
#print ("R^2 = ", R_squared)

# e
noise = norm.rvs(loc = 0, scale = 1, size = 30)  # normal
Y_noise = Y + noise

r2 = pearsonr(X,Y_noise)[0]
b2 = r2 * np.std(Y_noise) / X_sd
print("after noise : ")
print("r = ",r2)
print ("slope LS = ", b2)

# f , g

def calculate_r_b (Y,sdNoise, X_sd):
    noise = norm.rvs(loc=0, scale = sdNoise, size=30)  # normal
    Y = Y + noise

    r = pearsonr(X, Y)[0]
    b = r2 * np.std(Y) / X_sd

    return r,b

rVector = []
bVector = []
sdVector = []
i = 0.5
while (i <= 10):
    r, b = calculate_r_b(Y, i, X_sd)
    rVector.append(r)
    bVector.append(b)
    sdVector.append(i)
    i += 0.1

print ("rVector")
print(rVector)
print ("bVector")
print(bVector)

plt.scatter(sdVector , rVector, label='r (sd of noise)')
plt.ylabel("r")
plt.xlabel("sd")
plt.legend()
plt.show()

plt.scatter(sdVector , bVector, label='b (sd of noise)')
plt.ylabel("b")
plt.xlabel("sd")
plt.legend()
plt.show()

'''
# Q3


# c
path = "C:/Users/Erezd/OneDrive/Desktop/countries.csv"
df = pd.read_csv(path)

x1 = df['income']
x2 = df['education']
y = df['life_expectancy']
'''
# Graph 1:
plt.scatter(x1 , y)
plt.xlabel("income")
plt.ylabel("life expectancy")

# add least squares line :
A = np.vstack([x1, np.ones(len(x1))]).T
m, c = np.linalg.lstsq(A, y, rcond=None)[0]
plt.plot(x1, y, 'o', label='Ampirical data', markersize=10)
plt.plot(x1, m*x1 + c, 'b', label='least squares line')
plt.legend()
plt.show()


# Graph 2:
plt.scatter(x2 , y)
plt.xlabel("education")
plt.ylabel("life_expectancy")

# add least squares line :
A = np.vstack([x2, np.ones(len(x1))]).T
m, c = np.linalg.lstsq(A, y, rcond=None)[0]
plt.plot(x2, y, 'o', label='Ampirical data', markersize=10)
plt.plot(x2, m*x2 + c, 'b', label='least squares line')
plt.legend()
plt.show()

r1 = pearsonr(x1,y)[0]
R_squared_1 = r1 ** 2

r2 = pearsonr(x2,y)[0]
R_squared_2 = r2 ** 2

print (" (1) R^2 = ",R_squared_1)
print (" (2) R^2 = ",R_squared_2)

# log - x1
log_x1 = np.log(x1)
plt.scatter(log_x1 , y)
plt.xlabel("log( income )")
plt.ylabel("life expectancy")

A = np.vstack([log_x1, np.ones(len(x1))]).T
m, c = np.linalg.lstsq(A, y, rcond=None)[0]
plt.plot(log_x1, y, 'o', label='Ampirical data', markersize=10)
plt.plot(log_x1, m*log_x1 + c, 'b', label='log transformation - income')
plt.legend()
plt.show()
r3 = pearsonr(log_x1,y)[0]
R_squared_3 = r3 ** 2
print (" (log - income) R^2 = ",R_squared_3)

# log - x2
log_x2 = np.log(x2)
plt.scatter(log_x2 , y)
plt.xlabel("log( education )")
plt.ylabel("life expectancy")

A = np.vstack([log_x2 , np.ones(len(x1))]).T
m, c = np.linalg.lstsq(A, y, rcond=None)[0]
plt.plot(log_x2 , y, 'o', label='Ampirical data', markersize=10)
plt.plot(log_x2 , m*log_x2  + c, 'b', label='log transformation - education')
plt.legend()
plt.show()
r4 = pearsonr(log_x2 ,y)[0]
R_squared_4 = r4 ** 2
print (" (log - education) R^2 = ",R_squared_4)

# Q 4

r_midgami = np.cov(x2,y)[0][1] / (np.std(x2) * np.std(y))
print (r_midgami)

