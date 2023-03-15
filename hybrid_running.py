import numpy as np
import hybrid

print("____________EXTRACT____________")
predictions = np.load('../big_npy/resdata.npy',allow_pickle=True)
sent = np.load("../big_npy/testdata.npy",allow_pickle=True)
dictionary = np.load('../big_npy/lsa.npy',allow_pickle=True)
graph = np.load('../big_npy/graph.npy',allow_pickle=True)
graph_inverse = np.load('../big_npy/graph_inverse.npy',allow_pickle=True)
words_uniq = np.load('../big_npy/words_uniq.npy',allow_pickle=True)
n_occ = np.load('../big_npy/n_occ.npy',allow_pickle=True)
n_words = sum(n_occ)
dictionary = np.array(dictionary)
dictionary =  dictionary.tolist()



print("____________PREDICT____________")
sent = sent[:200]
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
            """try:
                print(predictions[i])
            except:
                print(len(predictions))
                print(len(sent))
                quit()"""
            b = hybrid.hybrid(predictions[i],sent[i][max(j-3,0):j],sent[i][j+1:min(j+4,len(sent[1])-1)],dictionary,graph,graph_inverse,words_uniq,n_occ,n_words,[1]*len(sent[i][max(j-3,0):j]+sent[i][j+1:min(j+4,len(sent[1])-1)]),1)
            predict[i] = b[0]
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
