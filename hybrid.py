import naive_bayes
import lsa
import numpy as np

def hybrid(predict,previos,next,dictionary,graph,graph_inverse,words_uniq,n_occ,n_words,lamd,alfa):
    ls = lsa.semantic_similarity(predict,previos,dictionary)
    nb = naive_bayes.test_data_make_prediction(graph,graph_inverse,words_uniq,n_occ,n_words,previos,next,lamd,predict)
    ls_sort = np.argsort(ls)
    nb_sort = np.argsort(nb)
    n_sum =  len(ls)*(len(ls)+1)/2
    for i in range(len(ls)):
        ls[ls_sort[i]] = (i+1)/n_sum
    for i in range(len(nb)):
        nb[nb_sort[i]] = (i+1)/n_sum
    ls_new = np.zeros(max(len(ls),len(nb)))
    nb_new = np.zeros(max(len(ls),len(nb)))
    for i in range(len(ls)):  
        ls_new[i] = ls[i]
    for i in range(len(nb)):  
        nb_new[i] = nb[i] 
    return alfa*nb_new + (1-alfa)*ls_new , nb_new, ls_new
