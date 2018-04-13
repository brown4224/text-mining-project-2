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

    query_vec = model[0].transform([query['query']])
    title_vec = model[0].transform([document['title']])
    cos = cosine_similarity(query_vec, title_vec)
    result = model[1].predict(cos)
    return result[0],0.5


def cosine_similarity(x,y):
    cos = cosine(x.toarray()[0], y.toarray()[0])
    if np.isfinite(cos):
        return cos
    return 0.0
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
    # vectorizer = CountVectorizer(max_df=0.7)
    vectorizer = TfidfVectorizer( max_df=1, lowercase=True, norm="l2")
    # vectorizer = TfidfVectorizer( max_df=0.3, stop_words="english", lowercase=True, norm="l2", ngram_range=(1, 2))
    vectorizer.fit(relevance_with_values["all_text"])




    ''' Step 4. Saving the model for TF features'''
    joblib.dump(vectorizer, 'resources/vectorizer.pkl')

    ''' Step 5. Converting query and title to vectors and finding cosine similarity of the vectors'''
    relevance_with_values["doc_vec"] = relevance_with_values.apply(lambda x: vectorizer.transform([x["title"] + x["body"]]), axis =1)
    relevance_with_values["query_vec"] = relevance_with_values.apply(lambda x: vectorizer.transform([x["query"]]), axis =1)
    relevance_with_values["cosine"]  = relevance_with_values.apply(lambda x: cosine_similarity(x['doc_vec'], x['query_vec']), axis=1)

    ''' Step 6. Defining the feature and label  for classification'''

    # X = [[relevance_with_values["cosine"], relevance_with_values["doc_vec"]]]
    # X = [[relevance_with_values["cosine"], relevance_with_values["doc_vec"]]]
    # X = relevance_with_values[["doc_vec"] ]
    X = relevance_with_values[["cosine"] ]
    Y = relevance_with_values["position"]

    print ("X: ", X.shape)
    print ("Y: ", Y.shape)


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