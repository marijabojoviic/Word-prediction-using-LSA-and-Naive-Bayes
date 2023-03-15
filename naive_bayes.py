import numpy as np

def binaryAdd (arr, l, r, x):
    # Check base case
    mid = l + (r - l)//2
    if r >= l:
        # If element is present at the middle itself
        try:
            if arr[mid][0] == x[0]:
                return mid , arr
        except:
            arr.append(x)
            return len(arr)-1,arr
        # If element is smaller than mid, then it can only
        # be present in left subarray
        if arr[mid][0] > x[0]:
            return binaryAdd(arr, l, mid-1, x) 

        # Else the element can only be present in right subarray
        else:
            return binaryAdd(arr, mid+1, r, x)
    else:
        arr.insert(l,x)
        return l, arr
def binarySearch (arr, l, r, x): 
     # Check base case
    mid = l + (r - l)//2
    if r >= l:
        # If element is present at the middle itself
        try:
            if arr[mid][0] == x[0]:
                return mid , arr
        except:
            arr.append(x)
            return len(arr)-1,arr
        # If element is smaller than mid, then it can only
        # be present in left subarray
        if arr[mid][0] > x[0]:
            return binaryAdd(arr, l, mid-1, x) 

        # Else the element can only be present in right subarray
        else:
            return binaryAdd(arr, mid+1, r, x)
    else:
        arr.insert(l,x)
        return l, arr

#make graph from list of sentences  
def make_graphs(sentences,depth):
    graph = []
    for d in range(depth):
        graph.append([])
        for ind,sentence in enumerate(sentences):
            sentence.reverse()
            if ind%1000 == 0:
                print(ind)
            for word in range(len(sentence)):
                index , graph[d] = binaryAdd(graph[d],0,len(graph[d]),[sentence[word]])
        #print(graph)
                if word - d - 1 >= 0:
                    k , graph[d] = binaryAdd(graph[d],0,len(graph[d]),[sentence[word-d-1]])
                    ii ,graph[d][k] = binaryAdd(graph[d][k],1,len(graph[d][k]),[sentence[word]])
                    #print(graph[d][k],ii)
                    if len(graph[d][k][ii]) == 1:
                        graph[d][k][ii].append(0)
                    graph[d][k][ii][1] +=1
    return graph 

#
def P_i(word_uniq,n_occ,word,total_n_word):
    for i in range(len(word_uniq)):
        if(word_uniq[i]==word):
            return n_occ[i]/total_n_word
    return 0

#
def gamma():
    return 1

#
def w_ij(graph,d,i,j):
    for ii in graph[d]:
        if i == ii[0]:
            for jj in ii[1:]:
                if jj[0] == j:
                    return jj[1]
    return 0

#
def Sw_ij(graph,d,i):
    s = 0.000000001
    
    for ii in graph[d]:
        if i == ii[0]:
            
            for jj in ii[1:]:
                s += jj[1]
            break
    return s

#
def P_ij(graph,d,i,j):
    
    k = w_ij(graph,d,i,j)/Sw_ij(graph,d,i)
    if k == 0:
        return 1
    else: 
        return k

#verovatnoca da je data rec sledeca rec
def B_j(graph,graph_inverse,j,prev,next,words_uniq,n_occ,total_n_word,lambd,):
    P = 1
    for i in range(len(prev)):
        P *= (P_ij(graph,i,prev[i],j)**lambd[i])
    for i in range(len(next)):
        P *= (P_ij(graph_inverse,i,next[i],j)**lambd[len(prev)+i])
    return P_i(words_uniq,n_occ,j,total_n_word)*P/gamma()


#
def normalization(x):
    u = len(x)*(len(x)+1)/2
    for i in range(len(x)):
        x[i][0] = (i+1)/u
    return x

def test_data_make_prediction(graph,graph_inverse,words_uniq,n_occ,n_word,prev,next,lambd,pred):
    lambd[0] = 1
    prev.reverse()
    for i in lambd[1:]:
        lambd[0] *= 1/ i
    x = np.ndarray((len(pred)))
    for ind, k in enumerate(pred):
        a = B_j(graph,graph_inverse,k,prev,next,words_uniq,n_occ,n_word,lambd)
        x[ind] = a
        
    return x


def train_B_j(graph,graph_inverse,j,prev,next,words_uniq,n_occ,total_n_word,lambd,):
    t = np.zeros(len(prev)+len(prev)+1)
    t += 1
    print("sta")
    for i in range(len(prev)):
        t[i] = P_ij(graph,i,prev[i],j)
    print("koji")
    for i in range(len(next)):
        print(len(prev)+i,len(t))
        t[len(prev)+i] = P_ij(graph_inverse,i,next[i],j)
        print("je")
    print("ovo")
    for i in range(len(t)):
        if t[i] == 0:
            t[i] = 1
    return P_i(words_uniq,n_occ,j,total_n_word),t



def train_data_make_prediction(graph,graph_inverse,words_uniq,n_occ,n_word,prev,next,lambd,pred,resenje):
    lambd[0] = 1
    prev.reverse()
    for i in lambd[1:]:
        lambd[0] *= 1/ i
    ver,t = train_B_j(graph,graph_inverse,pred[resenje],prev,next,words_uniq,n_occ,n_word,lambd)
    dlambd = train(t,lambd,gamma(),ver,0.1)
    for i in range(len(dlambd)):
        lambd[i+1] = lambd[i+1]*dlambd[i]
    print("%%%%%%%%%")
    print(lambd)
    print("%%%%%%%%%")

    return lambd

def train(t,lambd,g,ver,speed):
    dlambd = np.zeros(len(lambd)-1)
    for j in range(1,len(lambd)):
        t_multip = ver
        for i in range(len(lambd)):
            t_multip *= t[i]**lambd[i]
        print(t[j],lambd[j])
        t_multip /= t[j]**lambd[j]
        l_multip = 1
        for i in range(1,len(lambd)):
            l_multip *= lambd[i]
        dlambd[j-1] = ((1- (t_multip*t[j]**lambd[j]*
                        t[0]**(1/(l_multip)))/g)*
                        (t_multip*(t[j]**lambd[j])*np.log(t[0])*
                        (t[0]**(1/(l_multip)))/(g*l_multip*lambd[j])-
                        t_multip*t[j]**lambd[j]*np.log(t[j])*
                        (t[0]**(1/l_multip))/g))
    return dlambd*speed

def pisi():
    print("da li radis")

