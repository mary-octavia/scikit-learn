"""
==========================================
Spectral coclustering of e-mails and words
==========================================

Simultaneously cluster messages from the Twenty Newsgroups dataset and words.
On this application, coclustering performs better than MiniBatchKMeans in terms
of V-measure, using the newsgroup name as ground truth.

"""

from __future__ import print_function

from time import time
import re

import numpy as np

from sklearn.datasets.twenty_newsgroups import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.bicluster import SpectralCoclustering
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics.cluster import v_measure_score


def number_aware_tokenizer(doc):
    token_pattern = re.compile(u'(?u)\\b\\w\\w+\\b')
    tokens = token_pattern.findall(doc)
    tokens = ["#NUMBER" if token[0] in "0123456789_" else token
              for token in tokens]
    return tokens


vectorizer = TfidfVectorizer(stop_words='english', min_df=5,
                             tokenizer=number_aware_tokenizer)
cocluster = SpectralCoclustering(n_clusters=20, svd_method='arpack',
                                 random_state=0)
kmeans = MiniBatchKMeans(n_clusters=20, batch_size=5000, random_state=0)

twenty = fetch_20newsgroups()
y_true = twenty.target

print("Vectorizing...")
X = vectorizer.fit_transform(twenty.data)

print("Coclustering...")
start_time = time()
cocluster.fit(X)
y_cocluster = cocluster.row_labels_
print("Done in {}s. V-measure: {}".format(
    time() - start_time,
    v_measure_score(y_cocluster, y_true)))

print("MiniBatchKMeans...")
start_time = time()
y_kmeans = kmeans.fit_predict(X)
print("Done in {}s. V-measure: {}".format(
    time() - start_time,
    v_measure_score(y_kmeans, y_true)))

print("Most significant 10 words per cluster:")
feature_names = vectorizer.get_feature_names()

for cluster in xrange(20):
    cluster_docs = np.where(cocluster.row_labels_ == cluster)[0]
    cluster_words = np.where(cocluster.column_labels_ == cluster)[0]
    if not len(cluster_docs) or not len(cluster_words):
        continue
    out_of_cluster_docs = np.where(cocluster.row_labels_ != cluster)[0]
    word_col = X[:, cluster_words]

    word_scores = np.array(word_col[cluster_docs, :].sum(axis=0) -
                           word_col[out_of_cluster_docs, :].sum(axis=0))
    word_scores = word_scores.ravel()

    print(" ".join(feature_names[i] for i in word_scores.argsort()[:-10:-1]))
