import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

print('load data ...')
data = np.load('origin/1/data.npz')
t = np.array(data['t'])
X = np.array(data['X'])
Y = np.array(data['Y'])
Z = np.array(data['Z'])
print('diff ...')
t0 = np.append(np.diff(t), 0)
print('transform to df ...')
data = np.array([t0,X,Y,Z])
df = pd.DataFrame(data=data.T, columns=('t','X','Y','Z'))
print('statistic ...')
st = df.describe()
print('save result ...')
st.to_csv('statistic.csv')