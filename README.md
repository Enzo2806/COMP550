# Machine Translation Project - COMP 550 McGill University
## Authors
- Enzo Benoit-Jeannin (260969262)
- Marin Bergeron (26095497)
- Anthony Wilkinson (260966435)

## Project Overview
This repository contains the implementation of a machine translation system using Transformer models trained on the Europarl corpus for English to Italian and English to Spanish translations. The project was completed as part of the COMP 550 course at McGill University.

## Dataset
The datasets used for this project are not included in this repository due to their size. They can be downloaded from the [Europarl website](https://www.statmt.org/europarl/). After downloading, unzip the datasets into the `Dataset` folder of this repository with `en-it` and `en-es` subfolders for the English-Italian and English-Spanish datasets, respectively.

## Installation
To install the required dependencies, run the following command:
```bash
pip3 install -r requirements.txt
```

## Notes
As the model checkpoints were also very heavy, they were not included in this repository. Only the final fine-tuned model and the best english to italina baseline model are included.