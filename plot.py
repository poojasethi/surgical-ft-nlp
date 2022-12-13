import matplotlib.pyplot as plt

fig = plt.figure()

block = [1, 2, 3, 4, 5]

pos = [0.783, 0.791, 0.791, 0.788, 0.787]
ner = [0.841, 0.859, 0.864, 0.861, 0.856]
chunking = [0.815, 0.822, 0.825, 0.820, 0.802]
grammaticality = [0.78, 0.83, 0.84, 0.83, 0.80]

plt.plot(block, pos, label="pos")
plt.plot(block, ner, label="ner")
plt.plot(block, chunking, label="chunking")
plt.plot(block, grammaticality, label="grammaticality")

plt.xticks(block)
plt.xlabel("Block #")
plt.ylabel("Accuracy")

plt.legend(loc="upper left")
plt.title("Surgical Fine-Tuning Accuracy by Block")

plt.show()
