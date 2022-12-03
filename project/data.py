import torch

import pandas as pd
import logging
from transformers import BertTokenizer
from torch.utils.data import TensorDataset, random_split

logger = logging.getLogger(__name__)


def get_dataset(dataset: str):
    if dataset == "glue-cola":
        # Load the dataset into a pandas dataframe.
        df = pd.read_csv(
            "datasets/cola_public/raw/in_domain_train.tsv",
            delimiter="\t",
            header=None,
            names=["sentence_source", "label", "label_notes", "sentence"],
        )
        # Report the number of sentences.
        logger.info("Number of training sentences: {:,}\n".format(df.shape[0]))

        # Display 10 random rows from the data.
        logger.info(df.sample(10))

        # Get the lists of sentences and their labels.
        sentences = df.sentence.values
        labels = df.label.values

        logger.info("Loading BERT tokenizer")
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)

        def get_max_length():
            max_len = 0

            for sent in sentences:
                # Add `[CLS]` and `[SEP]` tokens
                input_ids = tokenizer.encode(sent, add_special_tokens=True)
                max_len = max(max_len, len(input_ids))

            logger.info("Max sentence length: ", max_len)

        max_len = get_max_length()

        # Load the BERT tokenizer.
        print("Loading BERT tokenizer...")
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)

        # TODO(pooja): Process the sentences in a batch rather than one-by-one.
        input_ids = []
        attention_masks = []

        for sentence in sentences:
            inputs = tokenizer.encode_plus(
                sentence,
                add_special_tokens=True,
                max_length=max_len,
                pad_to_max_length=True,
                return_tensor="pt",
            )
            input_ids.append(inputs["input_ids"])
            attention_masks.append(inputs["attention_mask"])

        input_ids = torch.cat(input_ids, dim=0)
        attention_masks = torch.cat(attention_masks, dim=0)
        labels = torch.tensor(labels)

        tensor_dataset = TensorDataset(input_ids, attention_masks, labels)

        # Create a 90-10 train-validation split.
        train_size = int(0.9 * len(tensor_dataset))
        val_size = len(tensor_dataset) - train_size
        train_dataset, val_dataset = random_split(tensor_dataset, [train_size, val_size])

        return train_dataset, val_dataset

    elif dataset == "glue-sst":
        raise NotImplementedError()
    elif dataset == "conll-pos":
        raise NotImplementedError()
    elif dataset == "conll-ner":
        raise NotImplementedError()
    else:
        raise ValueError(f"Unrecognized dataset {dataset}")
