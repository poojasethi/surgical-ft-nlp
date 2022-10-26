# Implementation of matplotlib function
import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure()
x = [1, 2, 4, 6, 8, 10]
protonet_accuracy = [0.987, 0.994, 0.995, 0.996, 0.997, 0.997]
protonet_error = [0.002, 0.001, 0.001, 0.001, 0.001, 0.001]
maml_accuracy = [0.976, 0.988, 0.993, 0.994, 0.993, 0.994]
maml_error = [0.003, 0.002, 0.001, 0.001, 0.001, 0.001]

plt.errorbar(x, protonet_accuracy, yerr=protonet_error, label="ProtoNet")
plt.errorbar(x, maml_accuracy, yerr=maml_error, label="MAML")

plt.legend(loc="upper left")

plt.title("ProtoNet and MAML accuracy")
plt.show()
