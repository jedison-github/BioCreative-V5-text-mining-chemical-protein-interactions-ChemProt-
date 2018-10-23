### BioCreative VI - Track 5: text mining chemical-protein interactions (ChemProt)
[http://www.biocreative.org/tasks/biocreative-vi/track-5/](http://www.biocreative.org/tasks/biocreative-vi/track-5/)


### Scripts usage

- [create_embeddings.py](create_embeddings.py) - To create pre-trained
PoS and dependency embedding vectors.

- [main.py](main.py) - To train a deep learning model (Bi-LSTM or CNN)
and test it. It is necessary to edit the script to choose different
input arguments. Only the seed number can be passed by command line
(e.g. `$ python main.py 0`).

- [mfuncs.py](mfuncs.py) - Convenient utilities used by `main.py`.

- [support.py](support.py) - Auxiliary utilities to treat the ChemProt
dataset.

- [utils.py](utils.py) - General use utilities.

- [voting.py](voting.py) - To average several outputs (probabilities).
Edit the script to choose the input directory and the group to be
evaluated.


### Data

- ChemProt dataset (sample, training, development, and test_gs) from
[http://www.biocreative.org/tasks/biocreative-vi/track-5/](http://www.biocreative.org/tasks/biocreative-vi/track-5/).

- BioGRID dataset from [https://thebiogrid.org/](https://thebiogrid.org/).

- The datasets were processed by the TEES software
[https://github.com/jbjorne/TEES](https://github.com/jbjorne/TEES)
(tokenization, sentence splitting, PoS tagging, and dependency parsing).


### word2vec embeddings

Word embedding models were created from PubMed English abstracts. They
can be downloaded from
[https://my.pcloud.com/publink/show?code=XZrkD97ZwkOrsOR9ffSDAgSBlQFFLh7hdS8k](https://my.pcloud.com/publink/show?code=XZrkD97ZwkOrsOR9ffSDAgSBlQFFLh7hdS8k).
Place them in the [word2vec](word2vec) directory.
Pre-trained PoS and dependency embedding vectors are provided in the
[word2vec](word2vec) directory.


### Requirements

Scripts were developed on Python version 3.6.4.

Required packages are (may work with other versions):
- gensim (3.4.0)
- Keras (2.1.6)
- matplotlib (2.2.2)
- numpy (1.14.3)
- networkx (2.1)
- scikit-learn (0.19.1)
- scipy (1.1.0)
- tensorflow (1.8.0)
