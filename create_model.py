import pandas as pd
from sklearn.feature_extraction.text import TfidfTransformer ,CountVectorizer, TfidfVectorizer
from scipy.spatial.distance import cosine
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.externals import joblib
from nltk import PorterStemmer
from nltk.corpus import stopwords
analyzer = CountVectorizer().build_analyzer()
import scipy
from sklearn.svm import SVC
import gensim, logging
from gensim.models.doc2vec import LabeledSentence

is_debug = False

'''Please feel free to modify this function (load_models)
   Make sure the function return the models required for the function below evaluate model
'''

def load_models():
    vectorizer = joblib.load('resources/vectorizer.pkl')
    clf = joblib.load('resources/classifier.pkl')
    return [vectorizer, clf]

'''Please feel free to modify this function (evaluate_model)
  Make sure the function only take three parameters: 1) Model (one or more model files), 2) Query, and 3) Document.
  The function always should return one of the positions (classes) and the confidence. In my case confidence is always 0.5.
  Preferably implement the function in a different file and call from here. Make sure keep everything else the same.
'''


def evaluate_model(model, query, document):
    query = query['query']
    title = document['title']
    body = document['body']
    query_vec = model[0].transform([query])
    title_vec = model[0].transform([title])
    body_vec = model[0].transform([body])

    cos_title = cosine_similarity(query_vec, title_vec)
    cos_body = cosine_similarity(query_vec, body_vec)

    common_title = common_terms(title, query)
    common_body = common_terms(body, query)

    # Query idf
    mq = np.max(query_vec)
    mqp = np.argmax(query_vec)
    lq = text_length(query)
    qs = np.sum(query_vec)
    qsn = qs / lq

    # Title idf
    mt = np.max(title_vec)
    mtp = np.argmax(title_vec)
    lt = text_length(title)
    ts = np.sum(title_vec)
    tsn = ts / lt

    # Body idf
    mb = np.max(body_vec)
    mbp = np.argmax(body_vec)
    lb = text_length(body)
    bs = np.sum(body_vec)
    bsn = bs / lb


    result = model[1].predict([[ cos_title, cos_body, common_title, common_body,
                                 mq, mqp, qs, qsn, lq,
                                 mt, mbp, bs, bsn, lb,
                                 mb, mbp, bs, bsn, lb]])




    # result = model[1].predict([[cos_title, common_title, cos_body, common_body]])
    return result[0],0.5


def common_terms(x,y):
    count = 0
    for token in set(x.split(" ")):
        if token in set(y.split(" ")):
            count += 1
    return count

def cosine_similarity(x,y, w2v=False):
    if x is None or y is None:
        return 0.0
    if w2v:
        cos = cosine(np.array(x), np.array(y))
    else:
        cos = cosine(x.toarray()[0], y.toarray()[0])


    if np.isfinite(cos):
        return cos
    return 0.0


def stemming(tokens):
    return (PorterStemmer().stem(token) for token in analyzer(tokens))


def text_length(X):
    return len(X) + 1


def idf_prob(doc_sparce):
    doc = scipy.sparse.coo_matrix(doc_sparce)
    dp = 1.0
    for doc_word, doc_idf in zip(doc.col, doc.data):
        dp *= doc_idf
    return dp


def convert_to_w2v(tokens, w2v):
    tokens = tokens.split(" ")
    pattern_vec = np.zeros(w2v.layer1_size)
    n_word = 1
    if len(tokens) > 1:
        for token in tokens:
            if token in w2v:
                pattern_vec = np.add(pattern_vec, w2v[token.strip()])
                n_word += 1
    pattern_vec = np.divide(pattern_vec, n_word)
    return pattern_vec.tolist()

# def convert_to_d2v(tokens, d2v):
#     pattern_vec = np.zeros(d2v.layer1_size)
#     tokens = tokens.split(" ")
#     n_word = 1
#     if len(tokens) > 1:
#         for token in tokens:
#             if token in d2v:
#                 pattern_vec = np.add(pattern_vec, d2v.infer_vector(token))
#                 n_word += 1
#     return np.divide(pattern_vec, n_word)
#     # np.array(d2v.infer_vector(tokens.split(" ")))




def create_model(all_documents_file, relevance_file,query_file):

    '''Step 1. Creating  a dataframe with three fields query, title, and relevance(position)'''
    documents = pd.read_json(all_documents_file)[["id", "title", "body"]]
    query_file = pd.read_json(query_file)[["query number","query" ]]

    relevance = pd.read_json(relevance_file)[["query_num", "position", "id"]]
    rv = relevance.merge(query_file,left_on ="query_num", right_on="query number")[ ["id","query", "position"]]\
        .merge(documents,left_on ="id", right_on="id") [["query", "position", "title", "body"]]


    '''Step 2. Creating  a column for creating index'''

    rv ["all_text"] = rv.apply( lambda x : x["query"] + x["title"] + x["body"] , axis =1)

    ''' Step 3. Creating a model for generating TF feature'''

    vectorizer = TfidfVectorizer( min_df=0.0, max_df=1.0, stop_words="english", lowercase=True, norm="l2", strip_accents='ascii')
    # vectorizer = TfidfVectorizer(use_idf=True, sublinear_tf=True,smooth_idf=True, min_df=0.0, max_df=1.0, stop_words="english", lowercase=True, norm="l2", strip_accents='ascii')
    vectorizer = vectorizer.fit(rv["all_text"])


    ''' Word to Vec Model'''
    w2v_model = gensim.models.Word2Vec( min_count=1, workers=4)
    w2v_model.build_vocab(rv["all_text"])


    # ''' Doc to Vec Model'''
    # sentences = []
    # for index, row in rv.iterrows():
    #     sentences.append(LabeledSentence(row["all_text"].split(), ['SENT %s' % index]))
    # d2v_model = gensim.models.Doc2Vec(sentences)

    ''' Step 4. Saving the model for TF features'''
    joblib.dump(vectorizer, 'resources/vectorizer.pkl')
    w2v_model.save('resources/w2v.model')
    # d2v_model.save('resources/d2v.model')

    ''' Step 5. Converting query and title to vectors and finding cosine similarity of the vectors'''
    rv["query_vec"] = rv.apply(lambda x: vectorizer.transform([x["query"]]), axis =1)
    rv["doc_vec_title"] = rv.apply(lambda x: vectorizer.transform([x["title"] ]), axis =1)
    rv["doc_vec_body"] = rv.apply(lambda x: vectorizer.transform([x["body"]]), axis =1)

    '''  COS and COMMON '''
    rv["cosine_title"]  = rv.apply(lambda x: cosine_similarity(x['doc_vec_title'], x['query_vec']), axis=1)
    rv["cosine_body"]  = rv.apply(lambda x: cosine_similarity(x['doc_vec_body'], x['query_vec']), axis=1)
    rv["common_title"] = rv.apply(lambda x: common_terms(x["title"], x["query"] ), axis =1)
    rv["common_body"] = rv.apply(lambda x: common_terms(x["body"], x["query"] ), axis =1)

    '''  Word 2V'''
    rv["query_vec_w2v"] = rv.apply(lambda x: convert_to_w2v(x["query"], w2v_model), axis=1)
    rv["doc_vec_w2v"] = rv.apply(lambda x: convert_to_w2v(x["title"], w2v_model), axis=1)
    rv["body_vec_w2v"] = rv.apply(lambda x: convert_to_w2v(x["body"], w2v_model), axis=1)

    ''' Cos W2V'''
    rv["cosine_title_w2v"]  = rv.apply(lambda x: cosine_similarity(x['doc_vec_w2v'], x['query_vec_w2v'], w2v=True), axis=1)
    rv["cosine_body_w2v"]  = rv.apply(lambda x: cosine_similarity(x['body_vec_w2v'], x['query_vec_w2v'], w2v=True), axis=1)






    '''  Doc 2V'''
    # rv["query_vec_d2v"] = rv.apply(lambda x: convert_to_d2v(x["query"], d2v_model), axis=1)
    # rv["doc_vec_d2v"] = rv.apply(lambda x: convert_to_d2v(x["title"], d2v_model), axis=1)
    # print(rv["doc_vec_d2v"] )
    # rv["body_vec_d2v"] = rv.apply(lambda x: convert_to_d2v(x["body"], d2v_model), axis=1)

    # ''' Cos W2V'''
    # rv["cosine_title_w2v"]  = rv.apply(lambda x: cosine_similarity(x['doc_vec_w2v'], x['query_vec_w2v'], w2v=True), axis=1)
    # rv["cosine_body_w2v"]  = rv.apply(lambda x: cosine_similarity(x['body_vec_w2v'], x['query_vec_w2v'], w2v=True), axis=1)




    ''' IDF Extraction'''
    rv["max_query_idf"] = rv.apply(lambda x: np.max(x["query_vec"]), axis=1)
    rv["max_pos_query_idf"] = rv.apply(lambda x: np.argmax(x["query_vec"]), axis=1)
    rv["sum_query_idf"] = rv.apply(lambda x: np.sum(x["query_vec"]), axis=1)
    rv["len_query_idf"] = rv.apply(lambda x: text_length(x["query"]), axis=1)
    rv["norm_query_idf"] = np.divide(rv["sum_query_idf"], rv["len_query_idf"])
    rv["prob_query_idf"] = rv.apply(lambda x: idf_prob(x["query_vec"]), axis =1)

    rv["max_title_idf"] = rv.apply(lambda x: np.max(x["doc_vec_title"]), axis =1)
    rv["max_pos_title_idf"] = rv.apply(lambda x: np.argmax(x["doc_vec_title"]), axis =1)
    rv["sum_title_idf"] = rv.apply(lambda x: np.sum(x["doc_vec_title"]), axis =1)
    rv["len_title_idf"] = rv.apply(lambda x: text_length(x["title"]), axis =1)
    rv["norm_title_idf"] = np.divide(rv["sum_title_idf"] ,rv["len_title_idf"] )
    rv["prob_title_idf"] = rv.apply(lambda x: idf_prob(x["doc_vec_title"]), axis =1)

    rv["max_body_idf"] = rv.apply(lambda x: np.max(x["doc_vec_body"]), axis =1)
    rv["max_pos_body_idf"] = rv.apply(lambda x: np.argmax(x["doc_vec_body"]), axis =1)
    rv["sum_body_idf"] = rv.apply(lambda x: np.sum(x["doc_vec_body"]), axis =1)
    rv["len_body_idf"] = rv.apply(lambda x: text_length(x["body"]), axis =1)
    rv["norm_body_idf"] = np.divide(rv["sum_body_idf"] ,rv["len_body_idf"] )
    rv["prob_body_idf"] = rv.apply(lambda x: idf_prob(x["doc_vec_body"]), axis =1)







    ''' Step 6. Defining the feature and label  for classification'''
    X = rv[ ["cosine_title"]+ ["cosine_title_w2v"]+["common_title"] + ["cosine_body"] + ["cosine_body_w2v"] + ["common_body"]
        + ["max_query_idf"]  + ["max_pos_query_idf"]  + ["norm_query_idf"] + ["len_query_idf"] + ["prob_query_idf"]
        + ["max_title_idf"]  + ["max_pos_title_idf"]  + ["norm_title_idf"] + ["len_title_idf"] + ["prob_title_idf"]
        + ["max_body_idf"]   + ["max_pos_body_idf"]   + ["norm_body_idf"]  + ["len_body_idf"]  + ["prob_body_idf"]]

    #
    # X = rv[ ["cosine_title"]+ ["common_title"] + ["cosine_body"] + ["common_body"]]



    Y = [v for k, v in rv["position"].items()]


    ''' Step 7. Splitting the data for validation'''
    X_train, X_test, y_train, y_test = train_test_split(    X, Y, test_size = 0.33, random_state = 42)

    ''' Step 8. Classification and validation'''
    target_names = ['1', '2', '3','4']
    # clf = MultinomialNB().fit(X_train, y_train)
    clf = RandomForestClassifier().fit(X_train, y_train)


    # clf = SVC().fit(X_train, y_train)

    print(classification_report(y_test,  clf.predict(X_test), target_names=target_names))

    ''' Step 9. Saving the data '''
    joblib.dump(clf, 'resources/classifier.pkl')




if __name__ == '__main__':
    create_model("resources/cranfield_data.json", "resources/cranqrel.json", "resources/cran.qry.json")