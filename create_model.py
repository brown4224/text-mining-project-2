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

    common_title = common_terms(title, query)
    common_body = common_terms(body, query)



    cos_title = cosine_similarity(query_vec, title_vec)
    cos_body = cosine_similarity(query_vec, body_vec)

    result = model[1].predict([[cos_title, common_title, cos_body, common_body]])
    return result[0],0.5

def common_terms(x,y):
    count = 0
    for token in set(x.split(" ")):
        if token in set(y.split(" ")):
            count += 1
    return count

def cosine_similarity(x,y):
    cos = cosine(x.toarray()[0], y.toarray()[0])
    if np.isfinite(cos):
        return cos
    return 0.0

def idf_mul(X):
    total = 1.0
    for x in X:
        np.mul
        total *= x

    print(total)
    return total

def stemming(tokens):
    return (PorterStemmer().stem(token) for token in analyzer(tokens))


def text_length(X):
    return len(X) + 1

def idf_sparce(doc_vec, query_vec):
    count = []
    doc_max_pos = []
    doc_max = []
    doc_min = []
    doc_sum = []

    query_max_pos = []
    query_max = []
    query_min = []
    query_sum = []



    for i in range(len(doc_vec)):

        doc = scipy.sparse.coo_matrix(doc_vec[i])
        query = scipy.sparse.coo_matrix(query_vec[i])

        for doc_word, doc_idf in zip(doc.col, doc.data):
            dmp = 0
            dm = 0.0
            ds = 0.0
            qmp = 0
            qm = 0.0
            qs = 0.0
            c = 0
            for query_word, query_idf in zip(query.col, query.data):
                if(doc_word == query_word):
                    if( doc_idf > dm):
                        dm = doc_idf
                        dmp = doc_word
                    if(query_idf > qm):
                        qm = query_idf
                        qmp = query_word
                    ds += doc_idf
                    qs += query_idf
                    c += 1
        count.append(c)
        doc_max_pos.append(dmp)
        doc_max.append(dm)
        doc_sum.append(ds / (c + 1))
        query_max_pos.append(qmp)
        query_max.append(qm)
        query_sum.append(qs / (c + 1))
    return count, doc_max_pos, doc_max, doc_sum, query_max_pos, query_max, query_sum


        # print(zip(doc.row, doc.col, doc.data))
        # exit(1)



    # for i, j, v in zip(doc.row, doc.col, doc.data):
    #     print( "%s" % (v))
    #     # print( "(%d, %d), %s" % (i, j, v))


def create_model(all_documents_file, relevance_file,query_file):

    '''Step 1. Creating  a dataframe with three fields query, title, and relevance(position)'''
    documents = pd.read_json(all_documents_file)[["id", "title", "body"]]
    query_file = pd.read_json(query_file)[["query number","query" ]]

    relevance = pd.read_json(relevance_file)[["query_num", "position", "id"]]
    relevance_with_values = relevance.merge(query_file,left_on ="query_num", right_on="query number")[ ["id","query", "position"]]\
        .merge(documents,left_on ="id", right_on="id") [["query", "position", "title", "body"]]


    '''Step 2. Creating  a column for creating index'''

    relevance_with_values ["all_text"] = relevance_with_values.apply( lambda x : x["query"] + x["title"] + x["body"] , axis =1)

    ''' Step 3. Creating a model for generating TF feature'''

    # vectorizer = TfidfVectorizer(min_df=0.0, max_df=1.0, stop_words="english", lowercase=True, norm="l2", strip_accents='ascii')
    vectorizer = TfidfVectorizer(min_df=0.0, max_df=1.0, stop_words="english", lowercase=True, norm="l2", strip_accents='ascii')
    vectorizer = vectorizer.fit(relevance_with_values["all_text"])


    ''' Step 4. Saving the model for TF features'''
    joblib.dump(vectorizer, 'resources/vectorizer.pkl')

    ''' Step 5. Converting query and title to vectors and finding cosine similarity of the vectors'''
    relevance_with_values["query_vec"] = relevance_with_values.apply(lambda x: vectorizer.transform([x["query"]]), axis =1)
    relevance_with_values["doc_vec_title"] = relevance_with_values.apply(lambda x: vectorizer.transform([x["title"] ]), axis =1)
    relevance_with_values["doc_vec_body"] = relevance_with_values.apply(lambda x: vectorizer.transform([x["body"]]), axis =1)
    relevance_with_values["cosine_title"]  = relevance_with_values.apply(lambda x: cosine_similarity(x['doc_vec_title'], x['query_vec']), axis=1)
    relevance_with_values["cosine_body"]  = relevance_with_values.apply(lambda x: cosine_similarity(x['doc_vec_body'], x['query_vec']), axis=1)
    relevance_with_values["common_title"] = relevance_with_values.apply(lambda x: common_terms(x["title"], x["query"] ), axis =1)
    relevance_with_values["common_body"] = relevance_with_values.apply(lambda x: common_terms(x["body"], x["query"] ), axis =1)

    relevance_with_values["max_title_idf"] = relevance_with_values.apply(lambda x: np.max(x["doc_vec_title"]), axis =1)
    relevance_with_values["max_pos_title_idf"] = relevance_with_values.apply(lambda x: np.argmax(x["doc_vec_title"]), axis =1)
    relevance_with_values["sum_title_idf"] = relevance_with_values.apply(lambda x: np.sum(x["doc_vec_title"]), axis =1)
    relevance_with_values["len_title_idf"] = relevance_with_values.apply(lambda x: text_length(x["title"]), axis =1)
    relevance_with_values["norm_title_idf"] = np.divide(relevance_with_values["sum_title_idf"] ,relevance_with_values["len_title_idf"] )

    relevance_with_values["max_body_idf"] = relevance_with_values.apply(lambda x: np.max(x["doc_vec_body"]), axis =1)
    relevance_with_values["max_pos_body_idf"] = relevance_with_values.apply(lambda x: np.argmax(x["doc_vec_body"]), axis =1)
    relevance_with_values["sum_body_idf"] = relevance_with_values.apply(lambda x: np.sum(x["doc_vec_body"]), axis =1)
    relevance_with_values["len_body_idf"] = relevance_with_values.apply(lambda x: text_length(x["body"]), axis =1)
    relevance_with_values["norm_body_idf"] = np.divide(relevance_with_values["sum_body_idf"] ,relevance_with_values["len_body_idf"] )

    relevance_with_values["max_query_idf"] = relevance_with_values.apply(lambda x: np.max(x["query_vec"]), axis =1)
    relevance_with_values["max_pos_query_idf"] = relevance_with_values.apply(lambda x: np.argmax(x["query_vec"]), axis =1)
    relevance_with_values["sum_query_idf"] = relevance_with_values.apply(lambda x: np.sum(x["query_vec"]), axis =1)
    relevance_with_values["len_query_idf"] = relevance_with_values.apply(lambda x: text_length(x["query"]), axis =1)
    relevance_with_values["norm_query_idf"] = np.divide(relevance_with_values["sum_query_idf"] ,relevance_with_values["len_query_idf"] )
    #
    # relevance_with_values["title_count"], relevance_with_values["title_max_pos"], relevance_with_values["title_max"], relevance_with_values["title_sum"], \
    # relevance_with_values["query_title_max_pos"], relevance_with_values["query_title_max"], relevance_with_values["query_title_sum"] =  idf_sparce(relevance_with_values["doc_vec_title"], relevance_with_values["query_vec"])
    # relevance_with_values["body_count"], relevance_with_values["body_max_pos"], relevance_with_values["body_max"], relevance_with_values["body_sum"], \
    # relevance_with_values["query_body_max_pos"], relevance_with_values["query_body_max"], relevance_with_values["query_body_sum"] =  idf_sparce(relevance_with_values["doc_vec_body"], relevance_with_values["query_vec"])


    ''' Step 6. Defining the feature and label  for classification'''
    # X = relevance_with_values[ ["cosine_title"] + ["cosine_body"] + ["common_title"] + ["common_body"] \
    # + ["query_title_max_pos"]+ ["query_title_max"]+ ["query_title_sum"] \
    # + ["query_body_max_pos"] + ["query_body_max"] + ["query_body_sum"] \
    # + ["title_count"] + ["title_max_pos"] +["title_max"] + ["title_sum"] \
    # + ["body_count"] + ["body_max_pos"] +["body_max"] + ["body_sum"]]

    X = relevance_with_values[ ["cosine_title"]+ ["common_title"] + ["cosine_body"] + ["common_body"]
        + ["max_query_idf"]  + ["max_pos_query_idf"] + ["sum_query_idf"] + ["norm_query_idf"] + ["len_query_idf"]
        + ["max_title_idf"]  + ["max_pos_title_idf"] + ["sum_title_idf"] + ["norm_title_idf"] + ["len_title_idf"]
        + ["max_body_idf"]   + ["max_pos_body_idf"]  + ["sum_body_idf"]  + ["norm_body_idf"]  + ["len_title_idf"] ]


    # X = relevance_with_values[ ["cosine_title"] + ["cosine_body"] + ["common_title"] + ["common_body"] \
    # + ["max_query_idf"]  + ["max_pos_query_idf"] + ["sum_query_idf"]   + ["len_query_idf"] + ["norm_query_idf"] \
    # + ["max_title_idf"] + ["max_pos_title_idf"] + ["sum_title_idf"] +  ["norm_title_idf"] + ["len_title_idf"] \
    # + ["max_body_idf"] + ["sum_body_idf"]+ ["norm_body_idf"] + ["len_title_idf"] \
    # + ["query_title_max_pos"]+ ["query_title_max"]+ ["query_title_sum"] \
    # + ["query_body_max_pos"] + ["query_body_max"] + ["query_body_sum"] \
    # + ["title_count"] + ["title_max_pos"] +["title_max"] + ["title_sum"] \
    # + ["body_count"] + ["body_max_pos"] +["body_max"] + ["body_sum"]]

    Y = [v for k, v in relevance_with_values["position"].items()]


    ''' Step 7. Splitting the data for validation'''
    X_train, X_test, y_train, y_test = train_test_split(    X, Y, test_size = 0.33, random_state = 42)

    ''' Step 8. Classification and validation'''
    target_names = ['1', '2', '3','4']
    # clf = MultinomialNB().fit(X_train, y_train)
    clf = RandomForestClassifier().fit(X_train, y_train)

    print(classification_report(y_test,  clf.predict(X_test), target_names=target_names))

    ''' Step 9. Saving the data '''
    joblib.dump(clf, 'resources/classifier.pkl')




if __name__ == '__main__':
    create_model("resources/cranfield_data.json", "resources/cranqrel.json", "resources/cran.qry.json")
