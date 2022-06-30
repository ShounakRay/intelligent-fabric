# @Author: shounak.ray
# @Date:   2022-06-29T23:24:32-07:00
# @Last modified by:   shounak.ray
# @Last modified time: 2022-06-30T00:23:16-07:00

from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np


def tfidf_vectorizer(raw_documents, **kwargs):
    """
    USAGE of TF-IDF – Bag of Words Vectorizer + Transformer:
    tfidf_vectorizer(raw_documents, input='content', max_features = None, use_idf = True, smooth_idf = True, sublinear_tf = True)
        > returns numpy array.
    """
    # Verify structure of raw_documents
    if type(raw_documents) not in [np.ndarray, np.array, list]:
        try:
            raw_documents = np.array(raw_documents)
        except Exception as e:
            print('Failed to convert text to np.array. Proceeding with caution...')

    if kwargs.get('input', 'content') == 'content':
        try:
            if type(raw_documents[0]) is not str:
                raise ValueError('Input was specified as content, but didn\'t get content.')
        except:
            raise ValueError('Failed to verify content-like structure of input. Aborted.')

    # max_features: int, default=None
    #   If not None, build a vocabulary that only consider the top max_features ordered by term frequency across the corpus.
    vectorizer = TfidfVectorizer(encoding='utf-8', decode_error='strict',
                                 strip_accents=None, lowercase=True, norm='l2', **kwargs)
    return vectorizer.fit_transform(raw_documents=raw_documents).toarray()


def _get_data(_CATS=['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']):
    from sklearn.datasets import fetch_20newsgroups
    return fetch_20newsgroups(subset='train', categories=_CATS, shuffle=True, random_state=42).data


# def _test():
#     _CATS = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']
#     return tfidf_vectorizer(_get_data(), input='content', max_features=None, use_idf=True, smooth_idf=True, sublinear_tf=True)

# EOF
