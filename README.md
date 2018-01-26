Doc2Vec
=====================================================
This repository contains an implementation of the doc2vec algorithm.

Contact me if you have any questions or want to use the code.

Prerequisites
--------------
- Python 3.6.3 or newer

Input file / folder structures:
----------------
This program requires specific folders and files to work:
```bash
├── documents
│   ├── doc_0.txt
│   ├── doc_1.txt
│   ├── ...
│   └── doc_n.txt
├── main.py
├── labels.json
└── .gitignore
```

Every document that should be taken into account has to be inside one directory
- default this directory is `documents/` but can be set to any folder relative to `main.py`
- each file should simply contain the plain text of the document

All labels have to inside a json file of the following form:
```json
{
  "doc_0.txt": "Amazon Invoice",
  "doc_1.txt": "News article",
  "...": "...",
  "doc_n.txt": "Amazon Invoice"
}
```
- default this file is `labels.json` but can be set to any file relative to `main.py`
- note that the file extension is also part of the key

Additional files:
-----------------
- logs will be saved into `doc2vec.log`
- a 2d graph for visual feedback will be saved into `graph.eps`
- a JSON containing the 10 most similar documents for every document will be saved into `most_similars.json`
- note that in this json a document should be most similar to itself to see if the systems acted as expected

Packages
-------------
All packages can be installed using pip
- numpy
- scikit-learn
- gensim
- matplotlib

Run Locally
-----------
- Clone the repo
- Run ``python main.py --doc_dir=documents/ --label_file=labels.json``


All hyperparameters can be set using parameters:
``python main.py --save_dir=stored_models/``

A list of all hyperparameters and their use can be found using:
``python main.py --help``
