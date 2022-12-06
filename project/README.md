# Environment Setup
```
conda create --name cs330-project python=3.9
conda activate cs330-project
pip install -r requirements.txt
```

# Training models
Example usage:

## Sequence Classification
```
python main.py -m sequence -d glue-cola -p group_1
```

## Token Classification
```
python main.py -m token -d conll-pos -p group_1 --epochs 1
```

# Datasets
* CoLA (Corpus of Linguistic Acceptability): https://nyu-mll.github.io/CoLA/cola_public_1.1.zip
* CoNLL-2003: https://huggingface.co/datasets/conll2003
