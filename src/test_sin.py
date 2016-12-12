import numpy as np
import matplotlib.pylab as plt
import matplotlib.animation as anima
import NeuralNet


def animate(i):
    NN.train([input_vector, true_outcomes], iterations=i)
    yar = NN.output_values
    ax1.clear()
    ax1.plot(input_vector,yar)

Fs = 100
f = 5
sample = 100
input_vector =  np.arange(100., sample+100, 0.1)
true_outcomes = np.tan(2 * np.pi * f * input_vector / Fs)
NN = NeuralNet.NeuralNet(len(input_vector), len(true_outcomes), 20)
fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)
ani = anima.FuncAnimation(fig, animate, interval=100)
plt.show()
