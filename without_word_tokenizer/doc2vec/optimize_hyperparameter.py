import multiprocessing
import argparse
import logging
import os
from time import time
import warnings
import pickle
import csv

import itertools
from gensim.utils import simple_preprocess
from gensim.models.doc2vec import TaggedDocument
from gensim.models.doc2vec import Doc2Vec

from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from util.load_data import load_dataset
from util.model_evaluation import get_metrics

warnings.filterwarnings("ignore")
# logging.basicConfig(level=logging.INFO, format='%(asctime) %(levelname)s % (message)s')


CORES = multiprocessing.cpu_count()


# parse commandline argument
parser = argparse.ArgumentParser("optimize_hyperparameters.py")
# parser.add_argument("--mode", help="available modes: optimize", required=True)
parser.add_argument("--train", help="train folder")
parser.add_argument("--test", help="test folder")
# parser.add_argument("--s", help="path to save model")
args = parser.parse_args()

if not (args.train and args.test):
    parser.error("Mode benchmark requires --train and --test")


def save_model(filename, clf):
    with open(filename, 'wb') as f:
        pickle.dump(clf, f)


def convert_to_tagged(data):
    result = [TaggedDocument(words=simple_preprocess(record[1]), tags=[record[0]]) for record in data]
    return result


def vec_for_learning(model, tagged_docs):
    targets, regressors = zip(*[(doc.tags[0], model.infer_vector(doc.words, steps=20)) for doc in tagged_docs])
    return regressors, targets


def get_model(corpus, params):
    transformer = Doc2Vec(vector_size=params["vector_size"], 
                          dm=params["dm"], 
                          dbow_words=params["dbow_words"], 
                          negative=params["negative"], 
                          min_count=params["min_count"], 
                          workers=CORES, 
                          window=params["window"], 
                          iter=params["iter"])

    transformer.build_vocab(corpus)
    return transformer


def report_log(row):
    log_file = 'report.csv'
    with open(log_file, 'a') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(row)


def grid_search(parameters, train_path, test_path):
    result = {'accuracy': None, 'precision': None, 'recall': None, 'f1_score': None}
    report_columns = list(parameters.keys()) + list(result.keys())
    report_log(report_columns)

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

    model_count = len(list(itertools.product(*parameters.values())))
    print("Start optimizing hyperparameters")
    for i, parameter_set in enumerate(itertools.product(*parameters.values())):
        print("\nTraining model %d/%d" % (i+1, model_count))
        params = dict(zip(parameters.keys(), parameter_set))
        t0 = time()
        transformer = get_model(train_tagged, params)

        transformer.train(train_tagged, total_examples=transformer.corpus_count, epochs=transformer.iter)

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

        result = get_metrics(y_test_label, y_pred)
        report_log(list(params.values())+list(result.values()))



if __name__ == '__main__':
    train_path = os.path.abspath(args.train)
    test_path = os.path.abspath(args.test)
    # model_path = os.path.abspath(args.s)
    parameters = {
        'vector_size': [300, 1024],
        'dm': [0],
        'dbow_words': [0, 1],
        'negative': [1, 2, 3, 4, 5],
        'min_count': [2, 5, 10, 20],
        'window': [2, 5, 10, 20],
        # 'sample': [0, 1e-2, 1e-5],
        # 'workers': [multiprocessing.cpu_count()],
        'iter': [10, 20, 50, 100]
    }

    grid_search(parameters, train_path, test_path)