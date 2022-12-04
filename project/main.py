import argparse
import logging
import random
import time
import os
import pickle
from typing import List, Dict, Any

import numpy as np
import torch
from transformers import AdamW, get_linear_schedule_with_warmup

from data import get_datasets, get_num_labels, get_test_dataloader, get_train_dataloader
from model import get_model, get_tunable_parameters
from utils import sequence_accuracy, format_time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

"""
CS 330 meta-learning project: Surgical fine-tuning of the classical NLP pipeline.

References:
-----------
[1] BERT Fine-tuning Tutorial with PyTorch: http://mccormickml.com/2019/07/22/BERT-fine-tuning/
[2] Fine-tuning BERT for named-entity-recognition: https://github.com/NielsRogge/Transformers-Tutorials/blob/master/BERT/Custom_Named_Entity_Recognition_with_BERT.ipynb
[3] CoLA Public: https://nyu-mll.github.io/CoLA/
[4] CoNLL-2003: https://huggingface.co/datasets/conll2003
"""

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model-type", required=True, choices=["sequence", "token"])
    parser.add_argument("-d", "--dataset", required=True, choices=["glue-cola", "glue-sst", "conll-pos", "conll-ner"])

    parser.add_argument(
        "-p",
        "--tunable-parameters",
        required=True,
        choices=["all", "group_1", "group_2", "group_3", "group_4", "group_5"],
    )
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--eps", type=float, help="Epsilon value for Adam", default=1e-8)
    parser.add_argument(
        "--epochs",
        type=int,
        help="The number of epochs to run fine-tuning for. The BERT authors recommend between [2, 4].",
        default=4,
    )
    return parser.parse_args()


def main(args: argparse.Namespace):
    num_labels = get_num_labels(args.dataset)
    model = get_model(args.model_type, num_labels)

    train_dataset, val_dataset, test_dataset = get_datasets(args.dataset)
    train_dataloader = get_train_dataloader(train_dataset)
    validation_dataloader = get_test_dataloader(val_dataset)

    train(
        model,
        args.model_type,
        train_dataloader,
        validation_dataloader,
        args.lr,
        args.eps,
        args.tunable_parameters,
        args.epochs,
    )


def train(
    model,
    model_type: str,
    train_dataloader,
    validation_dataloader,
    lr: float,
    eps: float,
    tunable_parameters: str,
    epochs: int,
):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    optimizer = AdamW(
        get_tunable_parameters(model, tunable_parameters),
        lr=lr,
        eps=eps,
    )
    model.to(device)

    # Total number of training steps is [number of batches] x [number of epochs].
    # (Note that this is not the same as the number of training samples).
    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    tunable_parameters = get_tunable_parameters(model, option="None")

    training_stats = []
    # Measure the total training time for the whole run.
    total_t0 = time.time()

    # For each epoch...
    for epoch_i in range(0, epochs):
        # ========================================
        #               Training
        # ========================================
        # Perform one full pass over the training set.
        logger.info("")
        logger.info("======== Epoch {:} / {:} ========".format(epoch_i + 1, epochs))
        logger.info("Training...")

        # Measure how long the training epoch takes.
        t0 = time.time()

        # Reset the total loss for this epoch.
        total_train_loss = 0.0

        # Dropout and BatchNorm layers behave differently during training
        # vs. test.
        # https://stackoverflow.com/questions/51433378/what-does-model-train-do-in-pytorch)
        model.train()

        # For each batch of training data...
        for step, batch in enumerate(train_dataloader):
            # Progress update every 40 batches.
            if step % 40 == 0 and not step == 0:
                # Calculate elapsed time in minutes.
                elapsed = format_time(time.time() - t0)

                logger.info("  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.".format(step, len(train_dataloader), elapsed))

            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            model.zero_grad()

            result = model(
                b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels, return_dict=True
            )

            loss = result.loss
            logits = result.logits
            total_train_loss += loss.item()

            # Perform a backward pass to calculate the gradients.
            loss.backward()

            # Clip the norm of the gradients to 1.0.
            # This is to help prevent the "exploding gradients" problem.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()  # Update model parameters
            scheduler.step()  # Update the learning rate

    # Calculate the average loss over all of the batches.
    avg_train_loss = total_train_loss / len(train_dataloader)

    # Measure how long this epoch took.
    training_time = format_time(time.time() - t0)

    logger.info("")
    logger.info("  Average training loss: {0:.2f}".format(avg_train_loss))
    logger.info("  Training epoch took: {:}".format(training_time))
    # ========================================
    #               Validation
    # ========================================
    # After the completion of each training epoch, measure our performance on
    # our validation set.

    logger.info("")
    logger.info("Running Validation...")

    t0 = time.time()

    # Put the model in evaluation mode--the dropout layers behave differently
    # during evaluation.
    model.eval()

    # Tracking variables
    total_eval_accuracy = 0
    total_eval_loss = 0

    for batch in validation_dataloader:
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)

        # Tell pytorch not to bother with constructing the compute graph during
        # the forward pass, since this is only needed for backprop (training).
        with torch.no_grad():

            # Forward pass, calculate logit predictions.
            # token_type_ids is the same as the "segment ids", which
            # differentiates sentence 1 and 2 in 2-sentence tasks.
            result = model(
                b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels, return_dict=True
            )

        # Get the loss and "logits" output by the model. The "logits" are the
        # output values prior to applying an activation function like the
        # softmax.
        loss = result.loss
        logits = result.logits

        # Accumulate the validation loss.
        total_eval_loss += loss.item()

        # Move logits, masks, and labels to CPU
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to("cpu").numpy()
        mask = b_input_mask.to("cpu").numpy()

        # Calculate the accuracy for this batch of test sentences, and
        # accumulate it over all batches.
        if model_type == "sequence":
            total_eval_accuracy += sequence_accuracy(logits, label_ids)
        elif model_type == "token":
            continue

    # Report the final accuracy for this validation run.
    avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)
    logger.info("  Batch Accuracy: {0:.2f}".format(avg_val_accuracy))

    # Calculate the average loss over all of the batches.
    avg_val_loss = total_eval_loss / len(validation_dataloader)

    # Measure how long the validation run took.
    validation_time = format_time(time.time() - t0)

    logger.info("  Validation Loss: {0:.2f}".format(avg_val_loss))
    logger.info("  Validation took: {:}".format(validation_time))

    # Record all statistics from this epoch.
    training_stats.append(
        {
            "epoch": epoch_i + 1,
            "Training Loss": avg_train_loss,
            "Valid. Loss": avg_val_loss,
            "Valid. Accur.": avg_val_accuracy,
            "Training Time": training_time,
            "Validation Time": validation_time,
        }
    )
    logger.info("")
    logger.info("Training complete!")
    logger.info("Total training took {:} (h:mm:ss)".format(format_time(time.time() - total_t0)))

    save_model(model, args.dataset, args.tunable_parameters, training_stats)


def save_model(model, dataset: str, parameter_group: str, training_stats: List[Dict[str, Any]]):
    output_dir = f"models/{dataset}/{parameter_group}/"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger.info(f"Saving model to {output_dir}")

    # Save trained model
    # Can be reloaded using from_pretrained()
    model_to_save = model.module if hasattr(model, "module") else model  # Take care of distributed/parallel training
    model_to_save.save_pretrained(output_dir)

    with open(os.path.join(output_dir, "training_stats.pickle"), "wb") as fh:
        pickle.dump(training_stats, fh, protocol=pickle.HIGHEST_PROTOCOL)

    with open(os.path.join(output_dir, "training_stats.txt"), "w") as fh:
        training_stats_text = [str(stat) for stat in training_stats]
        fh.write("\n".join(training_stats_text))


if __name__ == "__main__":
    args = parse_args()
    main(args)
