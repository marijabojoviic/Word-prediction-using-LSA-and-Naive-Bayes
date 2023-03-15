import os
import string

import nltk
import numpy as np
import sklearn.datasets as skd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import Normalizer

path = "../big_database"

files = []
# r=root, d=directories, f = files
for r, d, f in os.walk(path):
    for file in f:
        if '.txt' in file:
            files.append(os.path.join(r, file))

def preprocessing(files): #sredjivanje dokumenta
    data = []
    print(files)
    for f in files:
        file_ = open(f, errors='ignore')
        text = file_.read()
        sentences = nltk.sent_tokenize(text)
        table = str.maketrans('\n', ' ')
        sentences = [sentence.translate(table) for sentence in sentences]
        table1 = str.maketrans('', '', string.punctuation)
        sentences = [sentence.translate(table1) for sentence in sentences]
        sentences = [sentence.lower() for sentence in sentences]
        table2 = str.maketrans('', '', string.digits)
        sentences = [sentence.translate(table2) for sentence in sentences]
        data += sentences
        file_.close()
    return data

def used_sent(data): #proverava da li ima više od 4 non-stop reči u rečenici
    stop_words = ['a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an', 'and', 'any', 'are', "arent",
                  'as', 'at',
                  'be', 'because', 'been', 'before', 'being', 'below', 'between', 'both', 'but', 'by',
                  'can', "cant", 'cannot', 'could', "couldn't", 'did', "didn't", 'do', 'does', "doesnt", 'doing',
                  "dont", 'down', 'during',
                  'each', 'few', 'for', 'from', 'further',
                  'had', "hadnt", 'has', "hasn't", 'have', "havent", 'having', 'he', "hed", "hell", "hes", 'her',
                  'here', "heres",
                  'hers', 'herself', 'him', 'himself', 'his', 'how', "how's",
                  'i', "id", "ill", "im", "ive", 'if', 'in', 'into', 'is', "isnt", 'it', "it's", 'its', 'itself',
                  "lets", 'me', 'more', 'most', "mustnt", 'my', 'myself',
                  'no', 'nor', 'not', 'of', 'off', 'on', 'once', 'only', 'or', 'other', 'ought', 'our',
                  'ours' 'ourselves', 'out', 'over', 'own',
                  'same', "shant", 'she', "she'd", "shell", "shes", 'should', "shouldnt", 'so', 'some', 'such',
                  'than', 'that', "thats", 'the', 'their', 'theirs', 'them', 'themselves', 'then', 'there', "theres",
                  'these', 'they', "theyd",
                  "theyll", "theyre", "theyve", 'this', 'those', 'through', 'to', 'too', 'under', 'until', 'up', 'very',
                  'was', "wasnt", 'we', "wed", "well", "were", "weve", 'were', "werent", 'what', "whats", 'when',
                  "whens", 'where',
                  "wheres", 'which', 'while', 'who', "whos", 'whom', 'why', "whys", 'will', 'with', "wont", 'would',
                  "wouldnt",
                  'you', "youd", "youll", "youre", "youve", 'your', 'yours', 'yourself', 'yourselves',
                  'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'hundred', 'thousand']
    words1 = []
    used_sent = []

    for i in data:
        a = 0
        words1 = nltk.tokenize.word_tokenize(i)
        for j in words1:
            if j not in stop_words:
                a = a + 1
        if a >= 4:
            used_sent.append(i)
    return used_sent


def dictionary_make(used_sent): #pravljenje rečnika
    vectorizer = TfidfVectorizer(used_sent, use_idf=True)
    bag_of_words = vectorizer.fit_transform(used_sent)
    t = vectorizer.get_feature_names()  # terms
    svd = TruncatedSVD(n_components=250)#uzima se broj sto jer je on preporučen
    lsa = svd.fit_transform(bag_of_words.T)
    dictionary = {}
    for i, word in enumerate(t):
        dictionary[word] = lsa[i]

    return(dictionary)


def semantic_similarity(prediction,previos,dictionary):#second norm distance
    prediction_probability = []
    for i in range(len(previos)+1):
        prediction_probability.append([])
    previos_probability = [[0]]*len(previos)
    #print(prediction_probability)
    for key, value in dictionary.items():
        for ind,prev in enumerate(previos):
            if key == prev:
                previos_probability[ind] = value
    print(len(previos_probability[0]))
    for key,value in dictionary.items():
        for pred in prediction:
            if key == pred:
                for prev in range(len(previos)):
                    if previos_probability[prev][0] != 0:
                        prediction_probability[prev].append(1 / ((np.linalg.norm(np.subtract(value, previos_probability[prev]))) + 1))
                        print(1 / ((np.linalg.norm(np.subtract(value, previos_probability[prev]))) + 1))
                    else:
                        prediction_probability[prev].append(0)
                        print('jbg')
                    print(key)
                prediction_probability[-1].append(key)
                    #print(prediction_probability[:][-2:])
        if len(prediction_probability[-1]) == len(prediction):
            break
    probability = np.zeros((len(prediction_probability[-1])))
    for ind in range(len(prediction_probability)-1):
        probability += prediction_probability[ind]
    probability /= 3
    return probability

def norm_sem_sim(new_b): #normalizovana semantic similarity
    last = []
    for i in new_b:
        l = 1 / 3 * i[0]#koristimo 3 jer gledamo tri prethodne reči
        last.append([l, i[1]])
    last = sorted(last)
    return last

def normalization(x):
    u = len(x)*(len(x)+1)/2
    for i in range(len(x)):
        x[i][0] = (i+1)/u
    return x

def main(files):
    data = preprocessing(files)
    used_sentt = used_sent(data)
    dictionary = dictionary_make(used_sentt)
    print("nesto")
    np.save("../big_npy/lsa.npy",dictionary)
    print("lol")
    quit()
    prev = ["the","god","is"]
    dictionary = np.load("../big_npy/lsa.npy",a1llow_pickle=True)
    dictionary = np.array(dictionary)
    
    dictionary =  dictionary.tolist()
    print(type(dictionary))
    last = semantic_similarity(prev, dictionary)
    print(last[:10])
    return last
#main(files)