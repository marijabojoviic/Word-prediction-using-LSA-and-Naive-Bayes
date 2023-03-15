import os
import string
from nltk.tokenize import word_tokenize
from nltk import sent_tokenize
from string import digits
import numpy as np
#import lsa
import naive_bayes
print("____________FORMAT____________")
predictions = []
with open("../sentence-completion/testing_data.csv") as file:
    data = file.read()

data = data.split("\n")

del data[0]
del data[-1]

a = 0
for ind in range(len(data)):
    
    data[ind] = data[ind].split(',',1)[1]
    data[ind] = data[ind].split('.,')
    if len(data[ind]) < 2:
        data[ind] = data[ind][0].split('.",')
    pon = data[ind][1].split(',')
    
    if len(pon) != 5:
        pon = ["800","seven","\"60,000\"","\"1,200\"","twenty-one"]
        
    predictions.append(pon)
   
    data[ind] = data[ind][0]
print(len(data))
sent = []
for ind , text in enumerate(data):        
    print(ind,len(sent))
    if ind != len(sent):
        sent[ind-1] += " " + sent[ind]
        del sent[ind]
        print([sent[ind-1]])
        print(predictions[ind-1])
    sentences = sent_tokenize(text)
    table = str.maketrans('\n', ' ')
    sentences = [sentence.translate(table) for sentence in sentences]
    punc = string.punctuation.replace("_","")
    table1 = str.maketrans('', '', punc)
    sentences = [sentence.translate(table1) for sentence in sentences]
    sentences = [sentence.lower() for sentence in sentences]
    table2 = str.maketrans('', '', digits)
    sentences = [sentence.translate(table2) for sentence in sentences]
    sent += sentences
print(len(sent))
print(len(predictions))
for i in range(len(sent)):
    sent[i] = sent[i].split(" ")
np.save('../big_npy/resdata.npy',predictions)
np.save("../big_npy/testdata.npy",sent)
print("____________EXTRACT____________")
graph = np.load('../big_npy/graph.npy',allow_pickle=True)
graph_inverse = np.load('../big_npy/graph_inverse.npy',allow_pickle=True)
words_uniq = np.load('../big_npy/words_uniq.npy',allow_pickle=True)
n_occ = np.load('../big_npy/n_occ.npy',allow_pickle=True)
n_words = sum(n_occ)
predict = []

print("____________PREDICT____________")
sent = sent[:100]
predict = np.zeros((len(sent),5))
for i in range(len(sent)):
    print("###############")
    if i % 1 == 0:
        print("___",i,"___")
    for j in range(len(sent[i])):
        if sent[i][j]== '':
            del sent[i][j]
        print(sent[i][j])
        if sent[i][j] == '_____':
            print(sent[i][max(j-3,0):j])
            try:
                print(predictions[i])
            except:
                print(len(predictions))
                print(len(sent))
                quit()
            b=naive_bayes.test_data_make_prediction(graph,graph_inverse,words_uniq,n_occ,n_words,sent[i][max(j-3,0):j],sent[i][j+1:min(j+4,len(sent[1])-1)],[1]*6,predictions[i])
            predict[i] = b
            break
    print("###############")
np.save("../big_npy/prob.npy",predict)
print("____________END_OF_PREDICT____________")

predict = np.load('../big_npy/prob.npy',allow_pickle=True)
resenja = np.load('../big_npy/resenja.npy',allow_pickle=True)

print(type(predict),type(predict[0]))


brojac = 0
uk = 0
for ind in range(len(predict)):

    res = np.argmax(predict[ind])
    print(res, resenja[ind])
    if  res == resenja[ind]:
        brojac += 1
    uk += 1

print(brojac/uk)