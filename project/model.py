from transformers import (AdamW, BertConfig, BertForSequenceClassification,
                          BertForTokenClassification)


def get_model(model_type: str, num_labels: int = 2):
    model = None

    if model_type == "sequence":
        model = BertForSequenceClassification.from_pretrained(
            "bert-base-uncased",
            num_labels=num_labels,
        )
    elif model_type == "token":
        model = BertForTokenClassification.from_pretrained(
            "bert-base-uncased",
            num_labels=num_labels,
        )

    return model


def get_tunable_parameters(model, option="all"):
    """
    Get blocks of parameters to surgically fine-tune.
    """

    parameters = []

    if option == "none":
        pass
    elif option == "all":
        parameters.extend(model.bert.encoder.layer.parameters())
    elif option == "group_1":
        parameters.extend(model.bert.encoder.layer[0:4].parameters())
    elif option == "group_2":
        parameters.extend(model.bert.encoder.layer[2:6].parameters())
    elif option == "group_3":
        parameters.extend(model.bert.encoder.layer[4:8].parameters())
    elif option == "group_4":
        parameters.extend(model.bert.encoder.layer[6:10].parameters())
    elif option == "group_5":
        parameters.extend(model.bert.encoder.layer[8:12].parameters())

    if type(model) == BertForSequenceClassification:
        parameters.extend(model.classifier.parameters())
    else:
        raise NotImplementedError(f"Unknown model type {type(model)}")

    return parameters
