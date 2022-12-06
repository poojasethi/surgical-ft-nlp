import logging
from typing import Dict, List, Optional, Tuple

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler, Subset, TensorDataset, random_split
from transformers import BertTokenizer

from datasets import load_dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Label to assign to special tokens, i.e. [CLS] and [SEP].
SPECIAL_TOKEN_LABEL = -100

# Maximum sequence length of BERT model
MAX_BERT_LENGTH = 512


def get_train_dataloader(train_dataset: Subset, batch_size: int = 32) -> DataLoader:
    return DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=batch_size)


def get_test_dataloader(test_dataset: Subset, batch_size: int = 32) -> DataLoader:
    """
    Return a DataLoader to use for validation or testing.
    """
    return DataLoader(test_dataset, sampler=SequentialSampler(test_dataset), batch_size=batch_size)


def get_num_labels(dataset: str) -> int:
    """
    Returns the number of labels in the given dataset (plus one to account for special tokens).
    Ideally, this would be inferred from the dataset, but for convenience, we hardcode the number of labels here.
    """
    if dataset == "glue-cola":
        return 2
    elif dataset == "glue-sst":
        raise NotImplementedError()
    elif dataset == "conll-ner":
        return 9 + 1
    elif dataset == "conll-pos":
        return 47 + 1
    elif dataset == "conll-chunk":
        return 23 + 1
    else:
        raise ValueError(f"Unsupported dataset {dataset}")


def dataset_to_labels(dataset: str) -> Dict[int, str]:
    label_id_to_str = {}

    if dataset == "glue-cola":
        label_id_to_str = {0: "grammatical", 1: "not grammatical"}
    elif dataset == "glue-sst":
        raise NotImplementedError()
    elif dataset == "conll-pos":
        label_id_to_str = {
            0: '"',
            1: "''",
            2: "#",
            3: "$",
            4: "(",
            5: ")",
            6: ",",
            7: ".",
            8: ":",
            9: "``",
            10: "CC",
            11: "CD",
            12: "DT",
            13: "EX",
            14: "FW",
            15: "IN",
            16: "JJ",
            17: "JJR",
            18: "JJS",
            19: "LS",
            20: "MD",
            21: "NN",
            22: "NNP",
            23: "NNPS",
            24: "NNS",
            25: "NN|SYM",
            26: "PDT",
            27: "POS",
            28: "PRP",
            29: "PRP$",
            30: "RB",
            31: "RBR",
            32: "RBS",
            33: "RP",
            34: "SYM",
            35: "TO",
            36: "UH",
            37: "VB",
            38: "VBD",
            39: "VBG",
            40: "VBN",
            41: "VBP",
            42: "VBZ",
            43: "WDT",
            44: "WP",
            45: "WP$",
            46: "WRB",
        }

    elif dataset == "conll-ner":
        label_id_to_str = {
            0: "O",
            1: "B-PER",
            2: "I-PER",
            3: "B-ORG",
            4: "I-ORG",
            5: "B-LOC",
            6: "I-LOC",
            7: "B-MISC",
            8: "I-MISC",
        }

    elif dataset == "conll-chunk":
        label_id_to_str = {
            0: "O",
            1: "B-ADJP",
            2: "I-ADJP",
            3: "B-ADVP",
            4: "I-ADVP",
            5: "B-CONJP",
            6: "I-CONJP",
            7: "B-INTJ",
            8: "I-INTJ",
            9: "B-LST",
            10: "I-LST",
            11: "B-NP",
            12: "I-NP",
            13: "B-PP",
            14: "I-PP",
            15: "B-PRT",
            16: "I-PRT",
            17: "B-SBAR",
            18: "I-SBAR",
            19: "B-UCP",
            20: "I-UCP",
            21: "B-VP",
            22: "I-VP",
        }

    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    label_id_to_str[SPECIAL_TOKEN_LABEL] = "special token"

    return label_id_to_str


def get_max_length(sentences: List[str], tokenizer) -> int:
    max_len = 0

    for sent in sentences:
        # Add `[CLS]` and `[SEP]` tokens
        input_ids = tokenizer.encode(sent, add_special_tokens=True)
        max_len = max(max_len, len(input_ids))

    logger.info("Max sentence length: ", max_len)
    return max_len


def get_datasets(dataset: str) -> Tuple[Subset, Subset, Optional[Subset]]:
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

        max_len = get_max_length(sentences, tokenizer)

        # TODO(pooja): Process the sentences in a batch rather than one-by-one.
        input_ids = []
        attention_masks = []

        for sentence in sentences:
            inputs = tokenizer.encode_plus(
                sentence,
                add_special_tokens=True,
                max_length=max_len,
                pad_to_max_length=True,
                return_attention_mask=True,
                return_tensors="pt",
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
        return train_dataset, val_dataset, None

    elif dataset == "glue-sst":
        raise NotImplementedError()
    elif dataset == "conll-pos":
        datasets = load_dataset("conll2003")

        train_dataset = datasets["train"]
        validation_dataset = datasets["validation"]
        test_dataset = datasets["test"]

        logger.info("Loading BERT tokenizer")
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)

        train_dataset, validation_dataset, test_dataset = (
            create_token_level_dataset(train_dataset, "pos_tags", tokenizer),
            create_token_level_dataset(validation_dataset, "pos_tags", tokenizer),
            create_token_level_dataset(test_dataset, "pos_tags", tokenizer),
        )
        return train_dataset, validation_dataset, test_dataset

    elif dataset == "conll-ner":
        datasets = load_dataset("conll2003")

        train_dataset = datasets["train"]
        validation_dataset = datasets["validation"]
        test_dataset = datasets["test"]

        logger.info("Loading BERT tokenizer")
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)

        train_dataset, validation_dataset, test_dataset = (
            create_token_level_dataset(train_dataset, "ner_tags", tokenizer),
            create_token_level_dataset(validation_dataset, "ner_tags", tokenizer),
            create_token_level_dataset(test_dataset, "ner_tags", tokenizer),
        )

        return train_dataset, validation_dataset, test_dataset
    elif dataset == "conll-chunk":
        datasets = load_dataset("conll2003")

        train_dataset = datasets["train"]
        validation_dataset = datasets["validation"]
        test_dataset = datasets["test"]

        logger.info("Loading BERT tokenizer")
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)

        train_dataset, validation_dataset, test_dataset = (
            create_token_level_dataset(train_dataset, "chunk_tags", tokenizer),
            create_token_level_dataset(validation_dataset, "chunk_tags", tokenizer),
            create_token_level_dataset(test_dataset, "chunk_tags", tokenizer),
        )

        return train_dataset, validation_dataset, test_dataset
    else:
        raise ValueError(f"Unrecognized dataset {dataset}")


# Reference: https://discuss.huggingface.co/t/how-to-deal-with-differences-between-conll-2003-dataset-tokenisation-and-ber-tokeniser-when-fine-tuning-ner-model/11129/2
def create_token_level_dataset(dataset: Dataset, label_type: str, tokenizer: BertTokenizer) -> TensorDataset:
    # Extract tokens and attention_masks
    all_words = dataset["tokens"]
    all_word_labels = dataset[label_type]

    input_ids = []
    attention_masks = []
    labels = []

    # We need to re-tokenize the input to match the BertTokenizer.
    for words, word_labels in zip(all_words, all_word_labels):
        inputs = tokenizer.encode_plus(
            " ".join(words),
            add_special_tokens=True,
            max_length=MAX_BERT_LENGTH,
            truncation=True,
            padding="max_length",
            return_attention_mask=True,
            return_tensors="pt",
        )
        input_ids.append(inputs["input_ids"])
        attention_masks.append(inputs["attention_mask"])

        # Update the labels to match our new tokens
        token_labels = []
        tokens = []
        for word, label in zip(words, word_labels):
            word_tokens = tokenizer.tokenize(word)
            tokens.extend(word_tokens)
            token_labels.extend([label] * len(word_tokens))

        # Also make sure to add labels for the special tokens and any additional padding.
        num_padding_tokens = MAX_BERT_LENGTH - min(len(token_labels) + 2, MAX_BERT_LENGTH)
        padding_labels = [SPECIAL_TOKEN_LABEL] * num_padding_tokens if num_padding_tokens > 0 else []
        token_labels_with_special = [SPECIAL_TOKEN_LABEL] + token_labels + [SPECIAL_TOKEN_LABEL] + padding_labels
        labels.append(torch.LongTensor(token_labels_with_special)[None, :])

    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.cat(labels, dim=0)

    tensor_dataset = TensorDataset(input_ids, attention_masks, labels)

    return tensor_dataset
