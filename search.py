from readers import read_queries, read_documents
from random import randint
inverted_index = {}
import pandas as pd

from create_model import evaluate_model
def remove_not_indexed_toknes(tokens):
    return [token for token in tokens if token in inverted_index]

def remove_duplicates(documents):
    dist_doc_ids = []
    distinct_documents =[]
    for document in documents:
        if document['id'] not in dist_doc_ids:
            distinct_documents.append(document)
            dist_doc_ids.append(document['id'])
    return distinct_documents

def merge_two_postings(first, second):
    if first is None and second is None:
        return []
    elif second is None:
        return first
    elif first is None:
        return second
    else:
        first.extend(second)
        return remove_duplicates(first)


def merge_postings(indexed_tokens):
    first_list = inverted_index[indexed_tokens[0]]
    second_list = []
    for each in range(1, len(indexed_tokens)):
        second_list = inverted_index[indexed_tokens[each]]
        first_list = merge_two_postings(first_list, second_list)
    return first_list




def rank_documents(documents, query, model):
    resultset = []
    for document in documents:
        relevancy, confidence = evaluate_model(model,query,document)
        resultset.append( {"relevancy": relevancy,"confidence": confidence, "document": document })

    df = pd.DataFrame(resultset).sort_values(by=["relevancy", "confidence"], ascending=False)
    return df["document"].tolist()


def merge_postings_and_rank(indexed_tokens, query, model):
    unranked_documents = merge_postings(indexed_tokens)
    return rank_documents(unranked_documents, query, model)

def search_query(query,model):
    tokens = tokenize(str(query['query']))
    indexed_tokens = remove_not_indexed_toknes(tokens)
    if len(indexed_tokens) == 0:
        return []
    elif len(indexed_tokens) == 1:
        return inverted_index[indexed_tokens[0]]
    else:
        return merge_postings_and_rank(indexed_tokens, query, model)


def tokenize(text):
    return text.split(" ")


def add_token_to_index(token, doc_id):
    if token in inverted_index:
        current_postings = inverted_index[token]
        current_postings.append(doc_id)
        inverted_index[token] = current_postings
    else:
        inverted_index[token] = [doc_id]


def add_to_index(document):
    for token in tokenize(document['title']):
        add_token_to_index(token, document)


def create_index():
    for document in read_documents():
        add_to_index(document)
    print "Created index with size {}".format(len(inverted_index))


create_index()

if __name__ == '__main__':
    all_queries = [query for query in read_queries() if query['query number'] != 0]
    for query in all_queries:
        documents = search_query(query)
        print "Query:{} and Results:{}".format(query, documents)
