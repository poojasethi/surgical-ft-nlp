import matplotlib.pyplot as plt
from typing import List


def merge_lists(list2: List[float], list1: List[float], operator: str = "-") -> List[float]:
    assert len(list1) == len(list2)

    result = []
    for i in range(len(list1)):
        if operator == "-":
            result.append(list2[i] - list1[i])
        elif operator == "+":
            result.append(list2[i] + list1[i])
        else:
            raise ValueError(f"Unknown operator: {operator}")

    return result


labels = ["pos", "ner", "chunking", "grammaticality"]


class_head_results = [0.378, 0.681, 0.433, 0.71]
block_5_results = [0.787, 0.856, 0.802, 0.80]
block_3_results = [0.791, 0.864, 0.825, 0.84]
all_layer_results = [0.808, 0.868, 0.834, 0.84]

block_5_deltas = merge_lists(block_5_results, class_head_results, "-")
block_3_deltas = merge_lists(block_3_results, block_5_results, "-")
all_layer_deltas = merge_lists(all_layer_results, block_3_results, "-")

width = 0.35  # the width of the bars: can also be len(x) sequence

fig, ax = plt.subplots()

ax.bar(labels, block_5_deltas, width, label="Block 5")
ax.bar(labels, block_3_deltas, width, bottom=block_5_deltas, label="Block 3")
ax.bar(labels, all_layer_deltas, width, bottom=merge_lists(block_3_deltas, block_5_deltas, "+"), label="All Layers")

ax.set_ylabel("Absolute accuracy improvement")
ax.set_title("Accuracy improvement of surgical over classifier-head only fine-tuning")
ax.legend()

plt.show()
