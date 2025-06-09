import matplotlib.pyplot as plt
import numpy as np

with open("results.txt", 'r') as file:
    output = file.read().split('\n')
    file.close()

output = [output[i].split(' ')[2] for i in range(len(output))][:-1]
# print(type(output[0]))
output = list(map(eval, output))
output = np.array(output, dtype='float64')


vals, counts = np.unique(output, return_counts=True)

labels = np.ones(output[:21].shape)

labels = np.hstack((labels, -1*np.ones(output[21:].shape)))
accuracy = []
for threshold in np.arange(5, 15, 0.25):
    pred = np.zeros(output.shape)
    pred[output < threshold] = 1
    pred[output >= threshold] = -1
    accuracy.append((pred[pred == labels].shape[0])/pred.shape[0])
    print(threshold, accuracy[-1])

plt.plot(np.arange(5, 15, 0.25), accuracy)
plt.show()
