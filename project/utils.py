import datetime
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from seqeval.metrics import classification_report
from sklearn.metrics import accuracy_score


def sequence_accuracy(preds, labels):
    """
    Calculate sequence-level accuracy.
    """
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


def token_accuracy(preds, labels, mask, num_labels):
    """
    Calculate token-level accuracy.
    """
    # compute training accuracy
    active_logits = np.reshape(preds, (-1, num_labels))  # (batch_size * seq_len, num_labels)
    preds_flat = np.argmax(active_logits, axis=1)  # (batch_size * seq_len,)
    labels_flat = labels.flatten()  # (batch_size * seq_len,)

    # Use mask to determine where we should compare predictions with labels
    # (Includes [CLS] and [SEP] token predictions)
    active_accuracy = mask.flatten() == 1  # (batch_size * seq_len,)

    labels = labels_flat[active_accuracy]
    predictions = preds_flat[active_accuracy]

    return accuracy_score(labels, predictions), predictions, labels


def make_token_classification_report(all_preds, all_labels, label_id_to_str) -> str:
    """
    Calculate per-token metrics.
    """
    labels = [label_id_to_str[label_id] for label_id in all_labels]
    predictions = [label_id_to_str[label_id] for label_id in all_preds]
    return classification_report([labels], [predictions])


def format_time(elapsed):
    """
    Takes a time in seconds and returns a string hh:mm:ss
    """
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))

    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


def plot_training_stats(training_stats: Dict[str, Any]):
    # Display floats with two decimal places.
    pd.set_option("precision", 2)

    # Create a DataFrame from our training statistics.
    df_stats = pd.DataFrame(data=training_stats)

    # Use the 'epoch' as the row index.
    df_stats = df_stats.set_index("epoch")

    # A hack to force the column headers to wrap.
    df = df.style.set_table_styles([dict(selector="th", props=[("max-width", "70px")])])

    # Use plot styling from seaborn.
    sns.set(style="darkgrid")

    # Increase the plot size and font size.
    sns.set(font_scale=1.5)
    plt.rcParams["figure.figsize"] = (12, 6)

    # Plot the learning curve.
    plt.plot(df_stats["Training Loss"], "b-o", label="Training")
    plt.plot(df_stats["Valid. Loss"], "g-o", label="Validation")

    # Label the plot.
    plt.title("Training & Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.xticks([1, 2, 3, 4])

    plt.show()
