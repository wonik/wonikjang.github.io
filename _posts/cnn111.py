import numpy as np
import matplotlib.pyplot as plt

# y = w1 * x1 + w2 * x2 + b

n = 30
epoch = 5
np.random.seed(0)
# x = np.random.uniform(0, 5, n)
x = np.random.rand(n,2)
x.shape

#x = np.reshape( np.arange(60), (30,2))
#x.shape

d = np.random.uniform(0, 5, n)
d.shape

#d = 2 * x[:,0]

w = np.random.rand(1,2)
b = np.random.uniform(0.001, 0.002, 1)

alpha = 0.9

#fig1 = plt.figure()
#ax1 = fig1.add_subplot(111)
#ax1.plot(x, d, 'ro')

epdist = np.zeros((epoch,2))
wdata = np.zeros((epoch,2))
bdata = np.zeros((epoch,1))

for j in range(0,epoch):
        print(j)
        for i in range(0,n):
                y = x[i,:].dot(w.T) + b
                e = d[i] - y
                dw = alpha * e * x[i,:] # (1,2)
                db = alpha * e
                w += dw
                b += db
                # print(w, b)
        wdata[j,:] = w
        bdata[j,:] = b
        d1 = x.dot(w.T) + b
        dist = d1 - y
        dist1 = np.mean(abs(dist)) # MAE
        epdist[j,0] = j ; epdist[j,1] = dist1 # Matrix for epoch # and MAE

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(epdist[:,0], epdist[:,1])
ax.set_xlabel('Number of Epoch')
ax.set_ylabel('Mean Absolute Error')
ax.set_title('Trend of MAE')





# Regression visualization
import seaborn as sns; sns.set(color_codes=True)
ax = sns.regplot(x=x[:,0], y=d)


# Result visualizaion
y_pred = np.zeros((epoch, n))
for j in range(0, epoch):
    y_pred[j] = x.dot(wdata[j, :].T) + bdata[j]

import pandas as pd

num = np.repeat(np.arange(epoch), n)
x1 = np.tile(x[:,0],epoch)
y1 = np.concatenate(y_pred)
df = pd.DataFrame({'epoch':num ,'x':x1, 'y':y1})

sns.lmplot("x", "y", data=df, hue='epoch', fit_reg=True)



'''
#fig1 = plt.figure()
#ax1 = fig1.add_subplot(111)
for j in range(0, epoch):
    y_pred[j] = x.dot(wdata[j,:].T) + bdata[j]

    #plt.plot(x, d, 'ro')
    #plt.plot(x[:,0], y_pred[j])

    fit = np.polyfit(x[:, 0], y_pred[j], 1)
    fit_fn = np.poly1d(fit)

    ax1.plot(x[:,0], y_pred[j], 'yo', x, fit_fn(x), '--k')

#colormap = plt.cm.gist_ncar
#colors = [colormap(i) for i in np.linspace(0, 1,len(ax1.lines))]

fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
ax1.set_color_cycle(['red', 'green', 'blue', 'yellow'])
ax1.plot(x, d, 'ro')

for j in range(0, epoch):
    y_pred[j] = x.dot(wdata[j, :].T) + bdata[j]
    # plt.plot(x, d, 'ro')
    # plt.plot(x[:,0], y_pred[j])

    fit = np.polyfit(x[:, 0], y_pred[j], 1)
    fit_fn = np.poly1d(fit)

    ax1.plot(x[:,0], y_pred[j])
'''


