import numpy as np
import matplotlib.pylab as plt
import matplotlib.animation as anima
import NeuralNet


def animate(i):
    NN.train([input_vector, true_outcomes], iterations=i)
    yar = NN.output_values
    ax1.clear()
    ax1.plot(input_vector, yar)


Fs = 100
f = 5
sample = 100
input_vector = np.arange(100., sample+100, 0.1)
test_vector = np.arange(0., sample+100, 0.1)
true_outcomes = np.sin(2 * np.pi * f * input_vector / Fs)
NN = NeuralNet.NeuralNet(len(input_vector), len(true_outcomes), 20)
fig = plt.figure()
ax1 = fig.add_subplot(1, 1, 1)
ani = anima.FuncAnimation(fig, animate, interval=1)
plt.show()

outcome = NN.predict(test_vector)
plt.plot(test_vector, outcome)
