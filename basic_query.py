from pymongo import MongoClient
import json
from nltk.stem import WordNetLemmatizer
from index_constructor import tokenize
import math



class Query:


    def __init__(self):
        cluster = 'mongodb://localhost:27017/?readPreference=primary&appname=MongoDB%20Compass&directConnection=true&ssl=false'
        client = MongoClient(cluster)
        db = client.index
        self.db_tokens = db.tokens

    def start(self):
        print('Enter quit to end program')
        self.get_user_input()
        print("Bye!")

    def retrieve_urls(self, userInput):
        print(f'These URLs were found with the token: {userInput}')
        for posting in self.invertedIndex[userInput]:
            print(posting['url'])


    def get_user_input(self):
        userInput = input('Enter query: ')
        while (userInput != 'quit'):
            ## add lemmetization in m2
            self.process_input(userInput)
            userInput = input('Enter query (by word): ')
    
    def normalize(self, token_scores):
        squared_sum = 0
        for wt in token_scores:
            squared_sum += (wt * wt)
        normalized = math.sqrt(squared_sum)
        return normalized

    def process_input(self, userInput):
        count = 0
        lemmatize_user_input = tokenize(userInput)
        # list of all token data
        lemmatizer = WordNetLemmatizer()

        # every document that contains at least one of the query words, as well as the score of that document
        # doc_id: score
        document_scores = dict()

        # for each query token, all documents containing that query and the score (computed during index construction)
        # query_token: {doc_id: score}
        query_documents = dict()

        # query_token: query normalized score 
        query_scores = dict()
        query_tf = dict()
        terms_wt = []

        # doc_id: url
        doc_url = dict()

        # query frequency
        for user_query in lemmatize_user_input.keys():
            user_query = lemmatizer.lemmatize(user_query)
            query_scores[user_query] = dict()
            if user_query not in query_tf.keys():
                query_tf[user_query] = 0
            query_tf[user_query] += 1
            
            # allow easier access to term document postings list, using doc id as the key
            
         # for loop extracts pymongo pointer into a dictionary -> data, only runs loop once
    # Cosine Normalization for Query and Store all Query Documents
        for query_term, score in query_scores.items():
            input_data = self.db_tokens.find({"token": query_term})
            
            for data in input_data:
                # Query (q)
                term_tf_w = 1 + math.log(query_tf[query_term], 10)
                term_idf = data["postings"]['idf']
                wt = term_idf * term_tf_w
            
    
                query_scores[query_term]['tf-w'] = term_tf_w
                query_scores[query_term]['idf'] = term_idf
                query_scores[query_term]['wt'] = wt
                terms_wt.append(wt)
                # key is doc ID, doc I
                query_documents[query_term] = dict()
                for doc_id, posting in data["postings"].items():
                    if (doc_id != "idf"):
                        query_documents[query_term][doc_id] = posting['normalized']
                        if doc_id not in document_scores.keys():
                            document_scores[doc_id] = 0
                        doc_url[doc_id] = posting['url']


        # cosine normalization for query
        normalized_denominator = self.normalize(terms_wt)
        for term, score in query_scores.items():
            
            score['normalized'] = round(( score['wt'] / normalized_denominator), 5)
            del score['tf-w']
            del score['idf']
            del score['wt']



        for query_term, postings in query_documents.items():
            normalized_query = query_scores[query_term]['normalized']
            for doc_id, score  in postings.items():
                # score = Prod(q * d1) + Prod(q * d2) 
                document_scores[doc_id] += (normalized_query * score)
                document_scores[doc_id] = round(document_scores[doc_id], 5)
        sorted_scores = {k: v for k, v in sorted(document_scores.items(), key=lambda item: item[1], reverse=True)}


        print()
        print(f'Top 20 Documents for the query " {userInput} "')
        print()

        count = 1
        
        for doc_id, score in sorted_scores.items():
            print(f'{count}. {doc_url[doc_id]}')  
            print()
            count += 1
            if (count == 21):
                return

            
            
       



        
