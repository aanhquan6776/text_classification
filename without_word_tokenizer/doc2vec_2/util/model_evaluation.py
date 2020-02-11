import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt

def get_metrics(true_labels, predicted_labels):
    print('Accuracy: %0.1f%%' % (np.round(metrics.accuracy_score(true_labels, predicted_labels), 3)*100))
    print('Precision: %0.1f%%' % (np.round(metrics.precision_score(true_labels, predicted_labels, average='micro'), 3)*100))
    print('Recall: %0.1f%%' % (np.round(metrics.recall_score(true_labels, predicted_labels, average='micro'), 3)*100))
    print('F1 Score: %0.1f%%' % (np.round(metrics.f1_score(true_labels, predicted_labels, average='micro'), 3)*100))
