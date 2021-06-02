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
Place the tokenized data in path `data/<language_code/src.txt`.

## Get Polysemy Estimates

To get the embeddings, run the following command:
```bash
$ make get_embeddings LANGUAGE=en
```
The languages supported in this repository are: af, ar, bn, en, et, fa, fi, he, id, is, kn, ml, mr, pt, tl, tr, tt, yo.
Further, you can control the amount of used data with optional parameters `MAX_LINES`. In the paper, we used `1,000,000` sentences, but we left the default here to `100,000`. To increase it back, just run:
```bash
$ make get_embeddings LANGUAGE=en MAX_LINES=1000000
```

After getting the embeddings, merge them per word and get their covariances:
```bash
$ make merge_embeddings LANGUAGE=en
```

Finally, get a tsv with polysemy estimates by running:
```bash
$ make get_polysemy LANGUAGE=en
```
This file will be located in 'results/<lang>/polysemy_var.tsv' (for the polysemy estimates using variances) and 'results/<lang>/polysemy_cov.tsv' (for the polysemy estimates using the embeddings full covariance matrices).



## Get Surprisal Estimates

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


## Analysis


Finally, to analyse the results first merge polysemy and surprisals for all languages:
```bash
$ make merge_results LANGUAGE=en
```

Then run the following two scripts to analyse BERT and WordNet polysemys

```bash
$ python src/h04_analysis/print_bert_correlations.py
$ python src/h04_analysis/print_wordnet_correlations.py
```


## Extra Information

#### Citation

If this code or the paper were usefull to you, consider citing it:

```bash
@inproceedings{pimentel-etal-2020-speakers,
    title = "Speakers Fill Lexical Semantic Gaps with Context",
    author = "Pimentel, Tiago  and
      Hall Maudslay, Rowan  and
      Blasi, Damian  and
      Cotterell, Ryan",
    booktitle = "Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)",
    month = nov,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.emnlp-main.328",
    doi = "10.18653/v1/2020.emnlp-main.328",
    pages = "4004--4015",
}
```


#### Contact

To ask questions or report problems, please open an [issue](https://github.com/tpimentelms/lexical-ambiguity-in-context/issues).
