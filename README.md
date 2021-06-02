# lexical-ambiguity-in-context

This code accompanies the paper ["Speakers Fill Lexical Semantic Gaps with Context"](https://www.aclweb.org/anthology/2020.emnlp-main.328/) published in EMNLP 2020.

## Install

To install dependencies run:
```bash
$ conda env create -f environment.yml
```

And then install the appropriate version of pytorch, transformers and download spacy dependencies:
```bash
$ conda activate bert-ambiguity
$ conda install -y pytorch torchvision cudatoolkit=10.1 -c pytorch
$ # conda install pytorch torchvision cpuonly -c pytorch
$ pip install transformers
```

## Get wikipedia data

To get and tokenize data, use the github repository [tpimentelms/wiki-tokenizer](https://github.com/tpimentelms/wiki-tokenizer).

# Get embeddings

Get the embeddings:
```bash
$ make get_embeddings LANGUAGE=en
```

Merge embeddings per word and get their covariances:
```bash
$ make merge_embeddings LANGUAGE=en
```

Get a tsv with polysemy estimates:
```bash
$ make get_polysemy LANGUAGE=en
```
This file will be located in 'results/<lang>/polysemy_var.tsv' (for the polysemy estimates using variances) and 'results/<lang>/polysemy_cov.tsv' (for the polysemy estimates using the embeddings full covariance matrices).



# Get surprisals

Train bert to handle per word prediction:
```bash
$ make train_surprisal LANGUAGE=en
```

Get the surprisals with the trained model:
```bash
$ make get_surprisal LANGUAGE=en
```

Get the surprisals per word:
```bash
$ make merge_surprisal LANGUAGE=en
```


# Analysis


Finally, to analyse the results first merge polysemy and surprisals for all languages:
```bash
$ make merge_results LANGUAGE=en
```

Then run the following two scripts to analyse BERT and WordNet polysemys

```bash
$ python src/h04_analysis/print_bert_correlations.py
$ python src/h04_analysis/print_wordnet_correlations.py
```

# Make script for a language

Use the Makefile to do all steps above at once!
```bash
$ make LANGUAGE=en
```
