from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
from gensim import corpora, models
import gensim
import pickle
# import pyLDAvis.gensim
import os
from multiprocessing import Process, freeze_support
os.environ['MALLET_HOME'] = 'D:/UIBEResearch/mallet-2.0.8'

def main():
    # ======= begin configure ==========
    MAX_TOPICS = 8
    MIN_PERC = 0.8
    MAX_DS_SIZE = 100
    # ======== end configure ===========

    dict_symptoms = pickle.load(open("../datasets/covid19_symptoms.pickle", "rb"))
    list_categories = pickle.load(open("../datasets/covid19_tags.pickle", "rb"))

    tokenizer = RegexpTokenizer(r'\w+')

    # create English stop words list
    en_stop = get_stop_words('en')

    # Create p_stemmer of class PorterStemmer
    p_stemmer = PorterStemmer()

    # create sample documents

    # compile sample documents into a list
    # doc_set = [doc_a, doc_b, doc_c, doc_d, doc_e]
    doc_set = []
    doc_ids = []
    for key in dict_symptoms.keys():
        doc_set.append(dict_symptoms[key].split(","))
        doc_ids.append(str(key))

    # set max dataset
    import random

    doc_ids = random.sample(doc_ids, MAX_DS_SIZE)
    doc_set = [dict_symptoms[id].split(",") for id in doc_ids]

    # list for tokenized documents in loop
    texts = []

    # loop through document list
    for tokens in doc_set:
        # clean and tokenize document string

        # stem tokens
        # stemmed_tokens = [p_stemmer.stem(i) for i in tokens]

        # add tokens to list
        texts.append(tokens)

    # turn our tokenized documents into a id <-> term dictionary
    dictionary = corpora.Dictionary(texts)

    # convert tokenized documents into a document-term matrix
    corpus = [dictionary.doc2bow(text) for text in texts]

    mallet_path = 'D:/UIBEResearch/mallet-2.0.8/bin/mallet'  # update this path
    ldamallet = gensim.models.wrappers.LdaMallet(mallet_path, corpus=corpus, num_topics=MAX_TOPICS, id2word=dictionary)

    # Show Topics
    print(ldamallet.show_topics(formatted=False))

    # Compute Coherence Score
    from gensim.models import CoherenceModel

    coherence_model_ldamallet = CoherenceModel(model=ldamallet, texts=texts, dictionary=dictionary, coherence='c_v')
    coherence_ldamallet = coherence_model_ldamallet.get_coherence()
    print('\nCoherence Score: ', coherence_ldamallet)



if __name__ == '__main__':
    freeze_support()
    main()