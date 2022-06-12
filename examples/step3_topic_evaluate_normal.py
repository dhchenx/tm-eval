from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
from gensim import corpora, models
import gensim
import pickle
import pyLDAvis.gensim
from multiprocessing import Process, freeze_support
MAX_TOPICS=10
MIN_PERC=0.8

dict_symptoms=pickle.load(open("../datasets/covid19_symptoms.pickle","rb"))
list_categories=pickle.load(open("../datasets/covid19_tags.pickle","rb"))

tokenizer = RegexpTokenizer(r'\w+')

# create English stop words list
en_stop = get_stop_words('en')

# Create p_stemmer of class PorterStemmer
p_stemmer = PorterStemmer()

# create sample documents

# compile sample documents into a list
#doc_set = [doc_a, doc_b, doc_c, doc_d, doc_e]
doc_set=[]
doc_ids=[]
for key in dict_symptoms.keys():
    doc_set.append(dict_symptoms[key].split(","))
    doc_ids.append(str(key))

# set max dataset
import random
MAX_DS_SIZE=2000
doc_ids=random.sample(doc_ids,MAX_DS_SIZE)
doc_set=[dict_symptoms[id].split(",") for id in doc_ids]

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

# generate LDA model
#ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=MAX_TOPICS, id2word=dictionary, passes=20,alpha='auto',eta='auto')

# print keywords
#topics = ldamodel.print_topics(num_words=10)
#for topic in topics:
#    print(topic)

# ldamodel.save('model5.gensim')

list_tags=[]
for i in range(MAX_TOPICS):
    list_tags.append("t"+str(i+1))

from gensim.models import CoherenceModel
def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=3):
    """
    Compute c_v coherence for various number of topics

    Parameters:
    ----------
    dictionary : Gensim dictionary
    corpus : Gensim corpus
    texts : List of input texts
    limit : Max num of topics

    Returns:
    -------
    model_list : List of LDA topic models
    coherence_values : Coherence values corresponding to the LDA model with respective number of topics
    """
    coherence_values = []
    model_list = []
    for num_topics in range(start, limit, step):

        #model = gensim.models.wrappers.LdaMallet(mallet_path, corpus=corpus, num_topics=num_topics, id2word=id2word)
        model = gensim.models.ldamodel.LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=20,
                                                 )
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, corpus=corpus, coherence='c_v')
        c_score=coherencemodel.get_coherence()
        coherence_values.append(c_score)
        print(f"{num_topics}\t{c_score}")

    return model_list, coherence_values

import matplotlib.pyplot as plt
if __name__ == '__main__':
    freeze_support()
    # Show graph
    limit = 50
    start = 2
    step = 1
    print("Number of topics\tCoherence Value")
    model_list, coherence_values = compute_coherence_values(dictionary=dictionary, corpus=corpus, texts=texts,
                                                            start=start, limit=limit, step=step)
    x = range(start, limit, step)
    plt.plot(x, coherence_values)
    plt.xlabel("Num Topics")
    plt.ylabel("Coherence score")
    plt.legend(("coherence_values"), loc='best')
    plt.show()

    # Print the coherence scores
    for m, cv in zip(x, coherence_values):
        print("Num Topics =", m, " has Coherence Value of", round(cv, 4))
