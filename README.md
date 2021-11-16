# Multimodal Representation Learning!

## Requirements

-   Python 3.7+
-   Linux-based system

## Installation


>Clone this repository to your local machine.
```bash
git clone "git@github.com:goel-shashank/Multimodal-Representation-Learning.git"
cd "Multimodal-Representation-Learning"
```

### Environment Setup

Please follow the instructions at the following link to set up anaconda:
[Anaconda Setup](https://docs.anaconda.com/anaconda/install/index.html)

> Installing jupyter
```bash
$ pip install jupyter
```

The following commands create a conda environment inside the repository. The environment's kernel is added to jupyter notebook. 

> Set up the conda environment
```bash
$ DIR=${1:-.}
$ RDIR=$(realpath --relative-to="$HOME" $DIR)
$ conda env create --prefix $DIR/env -f environment.yml
$ source activate $DIR/env
$ $DIR/env/bin/pip install ipykernel
$ python -m pip install --upgrade pip
$ python -m ipykernel install --prefix $DIR/env --display-name "Python ($RDIR)"
$ conda deactivate
$ jupyter kernelspec install --user --name ${RDIR//\//$'-'} $DIR/env/share/jupyter/kernels/python3/
```

> Install the required python packages

```bash
$ conda activate ./env
$ pip install -r requirements.txt
```
---

## Dataset

We use the [Full English-WikiPedia Dataset](https://meta.wikimedia.org/wiki/Data_dump_torrents#English_Wikipedia) in this research.  The following commands can be used to download the dataset using torrent.

> Setting up rtorrent on linux (For more information: [Installing rtorrent](http://rakshasa.github.io/rtorrent))
```bash
$ cd
$ curl -o "libtorrent.tar.gz" "https://raw.githubusercontent.com/rakshasa/rtorrent-archive/master/libtorrent-0.13.8.tar.gz"
$ tar -xvzf "libtorrent.tar.gz"
$ cd "libtorrent-0.13.8"
$ ./configure --prefix="$HOME/local/" # Default prefix: "/use/local/"
$ make
$ make install
$ PKG_CONFIG_PATH="$HOME/libtorrent-0.9.8"
$ cd
$ rm -rf "libtorrent-0.13.8"
$ curl -o "rtorrent.tar.gz" "https://raw.githubusercontent.com/rakshasa/rtorrent-archive/master/rtorrent-9.tar.gz"
$ tar -xvzf "rtorrent.tar.gz"
$ cd "rtorrent-0.9.8"
$ ./configure --prefix="$HOME/local/" # Default prefix: "/use/local/"
$ make
$ make install
$ cd 
$ rm -rf "rtorrent-0.9.8"
```

> Downloading the Dataset (For more information: [Downloading Torrent using rtorrent](https://www.fosslinux.com/8688/how-to-download-torrents-using-the-command-line-in-terminal.htm) and [rtorrent cheatsheet](https://devhints.io/rtorrent))

```bash
$ cd $HOME
$ rtorrent
# Press enter
# Paste the following url: 
# https://www.litika.com/torrents/enwiki-20211020-pages-articles-multistream.xml.bz2.torrent
# Press enter
# Press up arrow key
# Ctrl + s
```

> After the download is complete, extract the dataset using the following command:

```bash
$ cd $HOME
$ bzip2 -dk "enwiki-20211020-pages-articles-multistream.xml.bz2"
```
