
# Multimodal Representation Learning!

## Requirements
 
- Python 3.7+
- Linux-based system

## Installation

>Clone this repository to your local machine.
```bash
git clone "git@github.com:goel-shashank/Multimodal-Representation-Learning.git"
cd  "Multimodal-Representation-Learning"
```

### Environment Setup

Please follow the instructions at the following link to set up anaconda:

[Anaconda Setup](https://docs.anaconda.com/anaconda/install/index.html)

The following commands create a conda environment inside the repository. 

> Set up the conda environment

```bash

$ DIR=${1:-.}
$ conda env create --prefix $DIR/env -f environment.yml
$ source activate $DIR/env
$ conda update --all
```

### WandB Login

> Login to wandb

```bash
$ wandb login
```

---

## Dataset

We use the [Conceptual Captions Dataset](https://meta.wikimedia.org/wiki/Data_dump_torrents#English_Wikipedia) in this research. The following commands can be used to download the dataset.

> Downloading the Conceptual Captions Dataset 
> 
```bash
$ TODO
```

---