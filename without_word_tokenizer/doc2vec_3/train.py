import argparse
import os
import pickle
from time import time
from gensim.utils import simple_preprocess
from gensim.models.doc2vec import TaggedDocument
from gensim.models.doc2vec import Doc2Vec
from sklearn.svm import LinearSVC

from util.load_data import load_dataset
from util.model_evaluation import get_metrics

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


parser = argparse.ArgumentParser("train.py")
parser.add_argument("--mode", help="available modes: train-test", required = True)
parser.add_argument("--train", help="train folder")
parser.add_argument("--test", help="test folder")
parser.add_argument("--s", help="path to save model")
args = parser.parse_args()


def save_model(filename, clf):
    with open(filename, 'wb') as f:
        pickle.dump(clf, f)


def convert_to_tagged(data):
    result = [TaggedDocument(words=simple_preprocess(record[1]), tags=[record[0]]) for record in data]

    return result


def vec_for_learning(model, tagged_docs):
    targets, regressors = zip(*[(doc.tags[0], model.infer_vector(doc.words, steps=20)) for doc in tagged_docs])
    return regressors, targets


if args.mode == "train-test":
    if not (args.train and args.test):
        parser.error("Mode train-test requires -- train and --test")
    if not args.s:
        parser.error("Mode train_test requires --s")
    train_path = os.path.abspath(args.train)
    test_path = os.path.abspath(args.test)

    print("Train model")
    model_path = os.path.abspath(args.s)
    print("Load data")
    X_train, y_train = load_dataset(train_path)
    X_test, y_test = load_dataset(test_path)
    train_corpus = [[y, x] for x, y in zip(X_train, y_train)]
    test_corpus = [[y, x] for x, y in zip(X_test, y_test)]
    
    train_tagged = convert_to_tagged(train_corpus)
    test_tagged = convert_to_tagged(test_corpus)

    target_names = list(set(y_train))

    print("%d documents (training set)" % len(X_train))
    print("%d documents (test set)" % len(X_test))
    print("%d categories" % len(target_names))
    print()

    print("Training model")
    t0 = time()

    transformer = Doc2Vec(vector_size=1024, dm=0, negative=5, min_count=2, workers=4, window=5, alpha=0.025, min_alpha=0.001, iter=400000)

    # total_epoch = 30
    transformer.build_vocab(train_tagged)

    transformer.train(train_tagged, total_examples=transformer.corpus_count, epochs=transformer.epochs)

    print("finish training doc2vec")

    X_train_vec, y_train_label = vec_for_learning(transformer, train_tagged)
    X_test_vec, y_test_label = vec_for_learning(transformer, test_tagged)

    print("start training classifier")
    model = LinearSVC()

    estimator = model.fit(X_train_vec, y_train_label)
    train_time = time() - t0
    print("train time: %dm %0.3fs" % (train_time/60, train_time - 60*(train_time//60)))

    t0 = time()
    y_pred = estimator.predict(X_test_vec)
    test_time = time() - t0
    print("test time: %dm %0.3fs" % (test_time/60, test_time - 60*(test_time//60)))	

    get_metrics(y_test_label, y_pred)
    save_model(os.path.join(model_path, 'transformer.pkl'), transformer)
    save_model(os.path.join(model_path, 'classifier.pkl'), estimator)

