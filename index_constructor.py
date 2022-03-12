from pymongo import MongoClient
import json
from bs4 import BeautifulSoup
import nltk
from nltk.stem import WordNetLemmatizer
import math

#import spacy
#https://spacy.io/api/token/#_title
# nlp = spacy.load("en_core_web_sm")
#   sentence = 'when i was in middle school i always wanted to be a computer science major because the department is terrific'
# doc = nlp(sentence)
# compounds = [doc[tok.i:tok.head.i + 1].text for tok in doc if tok.dep_ == "compound" and tok.tag_ != 'NNP']
# print(compounds)

STOP_WORDS = {"a", "able", "about", "above", "abst", "accordance", "according", "accordingly", "across", "act", "actually", "added", "adj", "affected", "affecting", "affects", "after", "afterwards", "again", "against", "ah", "all", "almost", "alone", "along", "already", "also", "although", "always", "am", "among", "amongst", "an", "and", "announce", "another", "any", "anybody", "anyhow", "anymore", "anyone", "anything", "anyway", "anyways", "anywhere", "apparently", "approximately", "are", "aren", "arent", "arise", "around", "as", "aside", "ask", "asking", "at", "auth", "available", "away", "awfully", "b", "back", "be", "became", "because", "become", "becomes", "becoming", "been", "before", "beforehand", "begin", "beginning", "beginnings", "begins", "behind", "being", "believe", "below", "beside", "besides", "between", "beyond", "biol", "both", "brief", "briefly", "but", "by", "c", "ca", "came", "can", "cannot", "can't", "cause", "causes", "certain", "certainly", "co", "com", "come", "comes", "contain", "containing", "contains", "could", "couldnt", "d", "date", "did", "didn't", "different", "do", "does", "doesn't", "doing", "done", "don't", "down", "downwards", "due", "during", "e", "each", "ed", "edu", "effect", "eg", "eight", "eighty", "either", "else", "elsewhere", "end", "ending", "enough", "especially", "et", "et-al", "etc", "even", "ever", "every", "everybody", "everyone", "everything", "everywhere", "ex", "except", "f", "far", "few", "ff", "fifth", "first", "five", "fix", "followed", "following", "follows", "for", "former", "formerly", "forth", "found", "four", "from", "further", "furthermore", "g", "gave", "get", "gets", "getting", "give", "given", "gives", "giving", "go", "goes", "gone", "got", "gotten", "h", "had", "happens", "hardly", "has", "hasn't", "have", "haven't", "having", "he", "hed", "hence", "her", "here", "hereafter", "hereby", "herein", "heres", "hereupon", "hers", "herself", "hes", "hi", "hid", "him", "himself", "his", "hither", "home", "how", "howbeit", "however", "hundred", "i", "id", "ie", "if", "i'll", "im", "immediate", "immediately", "importance", "important", "in", "inc", "indeed", "index", "information", "instead", "into", "invention", "inward", "is", "isn't", "it", "itd", "it'll", "its", "itself", "i've", "j", "just", "k", "keep", "keeps", "kept", "kg", "km", "know", "known", "knows", "l", "largely", "last", "lately", "later", "latter", "latterly", "least", "less", "lest", "let", "lets", "like", "liked", "likely", "line", "little", "'ll", "look", "looking", "looks", "ltd", "m", "made", "mainly", "make", "makes", "many", "may", "maybe", "me", "mean", "means", "meantime", "meanwhile", "merely", "mg", "might", "million", "miss", "ml", "more", "moreover", "most", "mostly", "mr", "mrs", "much", "mug", "must", "my", "myself", "n", "na", "name", "namely", "nay", "nd", "near", "nearly", "necessarily", "necessary", "need", "needs", "neither", "never", "nevertheless", "new", "next", "nine", "ninety", "no", "nobody", "non", "none", "nonetheless", "noone", "nor", "normally", "nos", "not", "noted", "nothing", "now", "nowhere", "o", "obtain", "obtained", "obviously", "of", "off", "often", "oh", "ok", "okay", "old", "omitted", "on", "once", "one", "ones", "only", "onto", "or", "ord", "other", "others", "otherwise", "ought", "our", "ours", "ourselves", "out", "outside", "over", "overall", "owing", "own", "p", "page", "pages", "part", "particular", "particularly", "past", "per", "perhaps", "placed", "please", "plus", "poorly", "possible", "possibly", "potentially", "pp", "predominantly", "present", "previously", "primarily", "probably", "promptly", "proud", "provides", "put", "q", "que", "quickly", "quite", "qv", "r", "ran", "rather", "rd", "re", "readily", "really", "recent", "recently", "ref", "refs", "regarding", "regardless", "regards", "related", "relatively", "research", "respectively", "resulted", "resulting", "results", "right", "run", "s", "said", "same", "saw", "say", "saying", "says", "sec", "section", "see", "seeing", "seem", "seemed", "seeming", "seems", "seen", "self", "selves", "sent", "seven", "several", "shall", "she", "shed", "she'll", "shes", "should", "shouldn't", "show", "showed", "shown", "showns", "shows", "significant", "significantly", "similar", "similarly", "since", "six", "slightly", "so", "some", "somebody", "somehow", "someone", "somethan", "something", "sometime", "sometimes", "somewhat", "somewhere", "soon", "sorry", "specifically", "specified", "specify", "specifying", "still", "stop", "strongly", "sub", "substantially", "successfully", "such", "sufficiently", "suggest", "sup", "sure", "t", "take", "taken", "taking", "tell", "tends", "th", "than", "thank", "thanks", "thanx", "that", "that'll", "thats", "that've", "the", "their", "theirs", "them", "themselves", "then", "thence", "there", "thereafter", "thereby", "thered", "therefore", "therein", "there'll", "thereof", "therere", "theres", "thereto", "thereupon", "there've", "these", "they", "theyd", "they'll", "theyre", "they've", "think", "this", "those", "thou", "though", "thoughh", "thousand", "throug", "through", "throughout", "thru", "thus", "til", "tip", "to", "together", "too", "took", "toward", "towards", "tried", "tries", "truly", "try", "trying", "ts", "twice", "two", "u", "un", "under", "unfortunately", "unless", "unlike", "unlikely", "until", "unto", "up", "upon", "ups", "us", "use", "used", "useful", "usefully", "usefulness", "uses", "using", "usually", "v", "value", "various", "'ve", "very", "via", "viz", "vol", "vols", "vs", "w", "want", "wants", "was", "wasnt", "way", "we", "wed", "welcome", "we'll", "went", "were", "werent", "we've", "what", "whatever", "what'll", "whats", "when", "whence", "whenever", "where", "whereafter", "whereas", "whereby", "wherein", "wheres", "whereupon", "wherever", "whether", "which", "while", "whim", "whither", "who", "whod", "whoever", "whole", "who'll", "whom", "whomever", "whos", "whose", "why", "widely", "willing", "wish", "with", "within", "without", "wont", "words", "world", "would", "wouldnt", "www", "x", "y", "yes", "yet", "you", "youd", "you'll", "your", "youre", "yours", "yourself", "yourselves", "you've", "z", "zero"}
invertedIndex = dict() # key = term, value = [docId, docId ...]
lemInvertedIndex = dict() # key = term, value = [docId, docId ...]
mongoIndex = list()
lemmatizer = WordNetLemmatizer()


# returns dictionary with tokens as the key, and frequency as the value
def tokenize(text):
    #Use token_dict to store frequency of each token in the input string
    token_dict = dict()
    token = ''  #Current token being read
    for char in text:
        if(not char.isascii()):
            #Ignore non-ASCII characters; this ensures that all non-English characters are excluded
            continue
        if(char.isalnum()):
            token += char.lower()
        #a non-alphanumeric character denotes the end of a token
        elif (token != ''):
            if (token not in STOP_WORDS):
                lemToken = lemmatizer.lemmatize(token)
                if (lemToken not in token_dict.keys()):
                    token_dict[lemToken] = 0
                token_dict[lemToken]+= 1
            token = ''
    #Be sure to process any tokens at the very end of the string
    if (token != ''):
        if (token not in STOP_WORDS):
            if (token not in token_dict.keys()):
                token_dict[token] = 0
            token_dict[token]+= 1
            token = ''

    return token_dict


#  {"token": {"postings": {docID: {"frequency": tf,"url":url, "tags": {"title": title_text}}}, docID: {"frequency": tf, "url":url, "tags": {"title": title_text}}} } }
def constructIndex(tokens, docId, url, tags):
    for token, frequency in tokens.items():
        posting = dict()
        posting[docId] = {"url": url, "frequency": frequency}
      #  posting = {"docID" {"frequency": tf, "url": url}}

        # add html tags to posting (if token is in the tag)
        if len(tags) != 0:
            html_tags = dict()
            for tag, token_string in tags.items():
                if (token in token_string):
                    html_tags[tag] = token_string
            # if token in the title, h1, h2, or h3 tags, a tags key to posting
            if len(html_tags) != 0:
                posting[docId]["tags"] = html_tags
           #     posting["tags"] = html_tags

        # add posting to an existing token or create new token with posting 
        if token in invertedIndex.keys():
          #  invertedIndex[token].append(posting)
            invertedIndex[token].update(posting)
        else:
           # invertedIndex[token] = [posting]
            invertedIndex[token] = posting


def add_tf_idf(corpusSize):
    
    for token, postings in invertedIndex.items():
      #  print(postings)
        document_frequency = len(postings)
        idf = round(math.log( (corpusSize/document_frequency), 10), 3)
        invertedIndex[token]['idf'] = idf
        for doc_id, posting in postings.items():
            if (doc_id != 'idf'):
            #for posting in postings:
                # raw term frequency,  how many times token is in the document
                tf = posting['frequency']
                # using log to weight the term frequency
                tf_wt = 1 + math.log(tf, 10)

                # increases with number of occurences within a document
                # increases with the rarity of the term in the collection
                # tf-idf
                wt = round(tf_wt * idf, 3)
               # posting['tf'] = round(tf, 3)
                posting['tf-idf'] = wt

                del posting['frequency']

def normalize(document_scores):
    squared_sum = 0
    for wt in document_scores:
        squared_sum += round((wt * wt), 3)
    normalized = round(math.sqrt(squared_sum), 3)
    return normalized

def normalize_vectors():
    for postings in invertedIndex.values():
        terms_wt = []
        for doc_id, posting in postings.items():
            if doc_id != 'idf':
                terms_wt.append(posting["tf-idf"])

        normalized_denominator = normalize(terms_wt)
        for doc_id, posting in postings.items():
            if doc_id != 'idf':
                posting['normalized'] = round(( posting["tf-idf"] / normalized_denominator), 3 )

    

def input_index_to_database():
    cluster = 'mongodb://localhost:27017/?readPreference=primary&appname=MongoDB%20Compass&directConnection=true&ssl=false'
    client = MongoClient(cluster)
    db = client.index
    index_db = db.tokens
    for token, postings in invertedIndex.items():
        d = {"token": token, "postings": postings}
        index_db.insert_one(d)

def important_tags_in_html(soup):
    tags = dict()
    if (soup.title is not None and soup.title.string != None):
        title = soup.title.string.strip()
        if (title != ""):
            lemmatized_title = ""
            tokenized_title = tokenize(title)
            for token in tokenized_title.keys():
                lemmatized_title += token
                lemmatized_title += " "
            if (lemmatized_title.strip() != ""):
                tags['title'] = lemmatized_title[:-1]
        
    if (soup.h1 is not None and soup.h1.string != None):
        header = soup.h1.string.strip()
        if (header != ""):
            lemmatized_header = ""
            tokenized_header = tokenize(header)
            for token in tokenized_header.keys():
                lemmatized_header += token
                lemmatized_header += " "
            if (lemmatized_header.strip() != ""):
                tags['h1'] = lemmatized_header[:-1]        

    if (soup.h2 is not None and soup.h2.string != None):
        header_two = soup.h2.string.strip()
        if (header_two != ""):
            lemmatized_header = ""
            tokenized_header = tokenize(header_two)
            for token in tokenized_header.keys():
                lemmatized_header += token
                lemmatized_header += " "
            if (lemmatized_header.strip() != ""):
                tags['h2'] = lemmatized_header[:-1]

    if (soup.h3 is not None and soup.h3.string != None):
        header_three = soup.h3.string.strip()
        if (header_three != ""):
            lemmatized_header = ""
            tokenized_header = tokenize(header_three)
            for token in tokenized_header.keys():
                lemmatized_header += token
                lemmatized_header += " "
            # tag contained only stop words
            if (lemmatized_header.strip() != ""):
                tags['h3'] = lemmatized_header[:-1]

    if (soup.strong is not None and soup.strong.string != None):
        strong = soup.strong.string.strip()
        if (strong != ""):
            lemmatized_strong = ""
            tokenized_strong = tokenize(strong)
            for token in tokenized_strong.keys():
                lemmatized_strong += token
                lemmatized_strong += " "
            if (lemmatized_strong.strip() != ""):
                tags['strong'] = lemmatized_strong[:-1]
    return tags


def getInvertedIndex():
    with open('WEBPAGES_RAW/bookkeeping.json') as f:
        data = json.load(f)
        corpusSize = len(data)
        path = 'WEBPAGES_RAW/'
        count = 0
        for docId, url in data.items():
            file = open(path + docId)
            htmlContent = ''
            for line in file:
                htmlContent += line 

# Process HTML tags <title>, <h1>, <h2>, <h3> into lemmatized string (to compare with token) and store in dictionary
            try:
                
                soup = BeautifulSoup(htmlContent, 'html.parser')
                # dict of all tags in html, using beautiful soup (which can handle broken html)
                html_tags = important_tags_in_html(soup)

                text = soup.get_text()
                

            except:
                print('error')

            tokens = tokenize(text)     
            constructIndex(tokens, docId, url, html_tags)
            count += 1
            
            print(count)
        add_tf_idf(corpusSize)
        normalize_vectors()
        input_index_to_database()
        
