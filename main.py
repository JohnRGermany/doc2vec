import argparse
import time
import datetime
import gensim
import os
import sys
import logging
import collections
import smart_open
import random
import glob
import re
import operator
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from accuracy import accuracy
from operator import itemgetter

def plot_tsne(vectors, labels, filenames, doc_dir):
    """Plots the TSNE-reduced 2D vectors and annotated labels.

    Parameters
    ----------
    vectors : [float] shape=(None, len(n))
        Vectors to be plotted.
    labels : [str] shape=(len(n,))
        A list of all labels
    filenames : str shape(n,)
        Names of the files corresponding to each vector
    doc_dir : str
        Relative directory to the documents

    Returns
    -------
    Nothing

    """
    logger.info('Preparing to plot result')
    fvs_std = StandardScaler().fit_transform(vectors)
    transformed = TSNE(n_components=2).fit_transform(fvs_std)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for doc_id, v in enumerate(transformed):
        ax.scatter(v[0], v[1], s=1)
        label_key = filenames[doc_id].replace(doc_dir, '')
        label = 'LABEL_NOT_FOUND'
        try:
            label = labels[label_key]
        except KeyError:
            logger.warning('No label found for file {0!s}'.format(label_key))
        ax.annotate(label, xy=v, xytext=v, size=0.5)

    plt.savefig("graph.eps", format='eps', dpi=1000)
    logger.info('Created plot in file graph.eps')

def get_labels(labels_file):
    """Reads the labels from disc.

    Parameters
    ----------
    labels_file : str
        Relative path to the json-file containing filename - label pairs

    Returns
    -------
    dict
        A python dict object contianing filename - label pairs

    """
    with open(labels_file) as f:
        labels = json.load(f)
        return labels

def vectorize_documents(model, filenames, vec_dir, train_corpus, num_features):
    """Creates vectors from documents.

    Parameters
    ----------
    model : gensim.models.doc2vec.Doc2Vec
        Trained model that receives documents as input and gives vectors as output.
    filenames : [str] shape=(None,)
        Names of the files corresponding to each vector.
    vec_dir : str
        Relative location of the folder where vectors are stored onto disc.
    train_corpus : [gensim.models.doc2vec.TaggedDocument] shape=(len(filenames))
        Contains all documents in required gensim format.
    num_features : int
        Number of features of output vectors

    Returns
    -------
    [float] shape=(None, len(filenames))
        Output vectors from model

    """
    vectors = np.empty((len(filenames), num_features))
    for doc_id, filename in enumerate(filenames):
        embedding = model.infer_vector(train_corpus[doc_id].words)
        vectors[doc_id] = embedding
        outfile_name = os.path.basename(filename) + ".npz"
        out_path = os.path.join(vec_dir, outfile_name)
        np.savetxt(out_path, embedding, delimiter=',')

    return vectors

def load_model(model_file):
    """Load a model from disc.

    Parameters
    ----------
    model_file : str
        Relative path to model.

    Returns
    -------
    gensim.models.doc2vec.Doc2Vec
        The Loaded model.

    """
    return gensim.models.doc2vec.Doc2Vec.load(model_file)

def save_model(model, save_dir):
    """Saves model to disc.

    Parameters
    ----------
    model : gensim.models.doc2vec.Doc2Vec
        Model to be saved.
    save_dir : str
        Relative storage directory.

    Returns
    -------
    Nothing

    """
    ts = time.time()
    t = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d_%H-%M-%S')
    if not os.path.exists(save_dir):
          os.makedirs(save_dir)
    path = os.path.join(save_dir, 'model_' + t)
    model.save(path)
    logger.info('Saved model to file {0!s}'.format(path))

def get_label(filename, labels, doc_dir):
    """Get label returns the label of a document.

    Parameters
    ----------
    filename : str
        Filename of the document
    labels : dict[str][str]
        Maps filenames and labels
    doc_dir : type
        Folder where the documents are saved

    Returns
    -------
    str
        Label

    """
    label = 'LABEL_NOT_FOUND'
    try:
        label = labels[filename.replace(doc_dir, '')]
    except KeyError:
        logger.warning('Did not find own label for file {0!s}'.format(filename))
    return label

def test_accuracy(model, test_corpus, n, labels, filenames, doc_dir):
    """Short summary.

    Parameters
    ----------
    model : gensim.models.doc2vec.Doc2Vec
        Pointer to model to be trained.
    train_corpus : [gensim.models.doc2vec.TaggedDocument] shape=(len(filenames),)
        Contains all documents in required gensim format.
    n : int
        Number of most similar documents
    labels : type
        Description of parameter `labels`.
    filenames : [str] shape=(None,)
        Names of the files corresponding to each vector.
    doc_dir : str
        Relative directory to the documents.

    Returns
    -------
    float
        A value that describes the accuracy of the model with 1 being best

    """
    t1 = datetime.datetime.now()
    logger.info('Starting accuracy test')
    data = []
    for d1_id in range(len(test_corpus)):
        d = {"label": get_label(filenames[d1_id], labels, doc_dir)}
        similarities = []
        for d2_id in range(len(test_corpus)):
            similarity = model.docvecs.similarity_unseen_docs(model, test_corpus[d1_id].words, test_corpus[d2_id].words)
            similarities.append((get_label(filenames[d2_id], labels, doc_dir), similarity))
        d["top_n"] = list(reversed(sorted(similarities, key=itemgetter(1))))[0:n]
        data.append(d)
    t2 = datetime.datetime.now()
    delta = t2 - t1
    logger.info('Finished accuracy test in {0!s} seconds'.format(delta.seconds))
    return accuracy(data)

def create_most_similar_json(model, train_corpus, n, labels, filenames, doc_dir):
    """Short summary.

    Parameters
    ----------
    model : gensim.models.doc2vec.Doc2Vec
        Pointer to model to be trained.
    train_corpus : [gensim.models.doc2vec.TaggedDocument] shape=(len(filenames),)
        Contains all documents in required gensim format.
    n : int
        Number of most similar documents
    labels : type
        Description of parameter `labels`.
    filenames : [str] shape=(None,)
        Names of the files corresponding to each vector.
    doc_dir : str
        Relative directory to the documents.

    Returns
    -------
    Nothing

    """
    logger.info('Creating similarity JSON file into most_similars.json')
    out_dict = {}
    for doc_id in range(len(train_corpus)):
        doc_dict = {}
        own_label = get_label(filenames[doc_id], labels, doc_dir)
        doc_dict["label"] = own_label
        inferred_vector = model.infer_vector(train_corpus[doc_id].words)
        sims = model.docvecs.most_similar([inferred_vector], topn=n)
        print(sims)
        similar_ids = [docid for docid, sim in sims]
        similarities = [sim for docid, sim in sims]
        similar_filenames = [filenames[i] for i in similar_ids]
        similar_labels = ["LABEL_NOT_FOUND"] * n
        for i in range(n):
            try:
                similar_labels[i] = labels[similar_filenames[i].replace(doc_dir, '')]
            except KeyError:
                logger.warning('No label found for file {0!s}'
                    .format(similar_filenames[i].replace(doc_dir, '')))
        similar_dicts = [
            {
                "filename": similar_filenames[i],
                "label": similar_labels[i],
                "similarity": similarities[i]
            }
            for i in range(n)
        ]
        doc_dict["most_similars"] = similar_dicts
        out_dict[filenames[doc_id]] = doc_dict

    with open('most_similars.json', 'w') as f:
        json.dump(out_dict, f)
    logger.info('Finished creating similarity JSON file')


def train_model(model, train_corpus):
    """Trains a model upon a corpus.

    Parameters
    ----------
    model : gensim.models.doc2vec.Doc2Vec
        Pointer to model to be trained.
    train_corpus : [gensim.models.doc2vec.TaggedDocument] shape=(len(filenames),)
        Contains all documents in required gensim format.

    Returns
    -------
    Nothing

    """
    logger.info('Start of model training')
    t1 = datetime.datetime.now()
    model.train(train_corpus, total_examples=model.corpus_count, epochs=model.iter)
    t2 = datetime.datetime.now()
    delta = t2 - t1
    logger.info('Finished training model in {0!s} seconds'.format(delta.seconds))

def read_corpus(filenames):
    """Yields a training corpus from filenames.

    Parameters
    ----------
    filenames : [str] shape=(n,)
        Names of files for training corpus

    Yields
    -------
    gensim.models.doc2vec.TaggedDocument
        Gensim representation of training document

    """
    for i, fname in enumerate(filenames):
        with smart_open.smart_open(fname, encoding="iso-8859-1") as f:
            s = ''.join([line for line in f])
            re.sub(r"\n", " ", s)
            if len(s) == 0 : logger.warning('File is empty {0!s}'.format(fname))
            yield gensim.models.doc2vec.TaggedDocument(gensim.utils.simple_preprocess(s), [i])

def create_model(doc_dir, num_features, num_iters, train_corpus):
    """Creates a doc2vec model.

    Parameters
    ----------
    doc_dir : str
        Relative directory to the documents.
    num_features : int
        Number of dimensions of inferred vectors.
    num_iters : int
        Number of iterations over the corpus.
    train_corpus : [gensim.models.doc2vec.TaggedDocument] shape=(len(filenames),)
        Contains all documents in required gensim format.

    Returns
    -------
    type
        Description of returned object.

    """
    model = gensim.models.doc2vec.Doc2Vec(vector_size=num_features, min_count=2, epochs=num_iters)
    model.build_vocab(documents=train_corpus)
    logger.info('Created model on {0!s} documents'.format(len(train_corpus)))

    return model

def run(FLAGS):
    """The main function of the program.

    Parameters
    ----------
    FLAGS : dictionary
        A dictionary containing all flags.

    Returns
    -------
    Nothing

    """
    try:
        assert os.path.exists(FLAGS.doc_dir)
    except AssertionError:
        logger.error('Documents folder not found: {0!s}'.format(FLAGS.doc_dir))
        sys.exit(0)
    try:
        assert os.path.exists(FLAGS.labels_file)
    except AssertionError:
        logger.error('Labels file not found: {0!s}'.format(FLAGS.labels_file))
        sys.exit(0)
    if not os.path.exists(FLAGS.vec_dir):
          os.makedirs(FLAGS.vec_dir)
    filenames = glob.glob(os.path.join(FLAGS.doc_dir, '*.txt'))
    try:
        assert len(filenames) > 0
    except AssertionError:
        logger.error('No documents in folder: {0!s}'.format(FLAGS.doc_dir))
        sys.exit(0)

    FLAGS.doc_dir = os.path.join(FLAGS.doc_dir, '')

    train_corpus = list(read_corpus(filenames))

    test_doc_dir = os.path.join('testdata', 'documents', '')
    test_labels_dir = os.path.join('testdata', 'labels.json')
    test_filenames = glob.glob(os.path.join(test_doc_dir, '*.txt'))
    test_labels = get_labels(test_labels_dir)
    test_corpus = list(read_corpus(test_filenames))

    if FLAGS.model_file == '':
        model = create_model(FLAGS.doc_dir, FLAGS.num_features, FLAGS.num_iters, train_corpus)
    else:
        model = load_model(FLAGS.model_file)

    if FLAGS.training != 0: train_model(model, train_corpus)
    if FLAGS.save_dir != '':
        save_model(model, FLAGS.save_dir)
    vectors = vectorize_documents(model, filenames, FLAGS.vec_dir, train_corpus, FLAGS.num_features)
    labels = get_labels(FLAGS.labels_file)
    n = 10 if len(filenames) >= 10 else len(filenames)
    # model.delete_temporary_training_data(keep_doctags_vectors=False, keep_inference=False)
    if FLAGS.plot != 0:
        plot_tsne(vectors, labels, filenames, FLAGS.doc_dir)
    accuracy = test_accuracy(model, test_corpus, n, test_labels, test_filenames, test_doc_dir)
    logger.info('Accuracy: {0!s}'.format(accuracy))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
      '--doc_dir',
      type=str,
      default='documents/',
      help='Directory containing documents.'
    )
    parser.add_argument(
      '--vec_dir',
      type=str,
      default='doc_vectors/',
      help='Directory to store document vectors in.'
    )
    parser.add_argument(
      '--labels_file',
      type=str,
      default='labels.json',
      help='File containing (document - label) pairs.'
    )
    parser.add_argument(
      '--num_iters',
      type=int,
      default=15,
      help='Number of training iterations.'
    )
    parser.add_argument(
      '--num_features',
      type=int,
      default=32,
      help='Dimensions of embeddings.'
    )
    parser.add_argument(
      '--save_dir',
      type=str,
      default='',
      help='Directory to store model after training. Does not save if empty.'
    )
    parser.add_argument(
      '--model_file',
      type=str,
      default='',
      help='Loads the model from disc if set. If not trains a new model.'
    )
    parser.add_argument(
      '--training',
      type=int,
      default=1,
      help='Whether or not the model should be trained. 0 for no training.'
    )
    parser.add_argument(
      '--plot',
      type=int,
      default=0,
      help='Whether or an ouput plot is created. 0 for no plotting.'
    )
    FLAGS, unparsed = parser.parse_known_args()
    logger = logging.getLogger('doc2vec')
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler('doc2vec.log', mode='a')
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    ts = time.time()
    logger.info('Starting new doc2vec instance')
    if len(unparsed) != 0:
        logger.warning('Unknown arguments passed: {0!s}'.format(unparsed))
    run(FLAGS)
