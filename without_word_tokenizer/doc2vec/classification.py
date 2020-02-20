import argparse
import os
import pickle
from os.path import dirname, join

from util.load_data import normalize_text
from gensim.utils import simple_preprocess
from time import time

parser = argparse.ArgumentParser("classification.py")
text = parser.add_argument_group("The following arguments are mandatory for text option")
text.add_argument("--text", metavar="TEXT", help="text to predict", nargs="?")
args = parser.parse_args()

path = join(dirname(__file__), "models")

transformer = pickle.load(open(join(path, "transformer.pkl"), 'rb'))
estimator = pickle.load(open(join(path, "classifier.pkl"), 'rb'))

if not args.text:
    parser.print_help()


if args.text:
    t0 = time()
    text = args.text
    text = normalize_text(text)
    print(text)
    X = transformer.infer_vector(simple_preprocess(text), steps=20)
    y = estimator.predict(X.reshape(1, -1))[0]
    classify_time = time() - t0
    print(y)
    print("process time: %0.3fs" % classify_time)
    
    