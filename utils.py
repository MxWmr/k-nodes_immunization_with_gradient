import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random as rd
import retworkx as rx


def von_mises(A, vector =False, eps =0.1, itemax =100):
    # Ideally choose a random vector
    # To decrease the chance that our vector
    # Is orthogonal to the eigenvector
    b_k = np.random.rand(A.shape[0])

    b_k1_norm = np.linalg.norm(b_k)
    v=0
    ite=0
    while abs(v-b_k1_norm)>eps and ite<itemax:
        v = b_k1_norm

        # calculate the matrix-by-vector product Ab
        b_k1 = A.dot(b_k)


        # calculate the norm
        b_k1_norm = np.linalg.norm(b_k1,ord=2)

        # re normalize the vector

        b_k = 1/(b_k1_norm+0.001)*b_k1

        ite+=1

    r_spec = (b_k.T).dot(A.dot(b_k))/(np.dot(b_k.T,b_k)+0.0001)

    if vector :
        return r_spec,b_k
    else:
        return r_spec


## Measure of vulnerability using spectral radius
def max_ev(G=None,A=None,vector=False):
    if A is None:
        A = rx.adjacency_matrix(G)
    r_spec = von_mises(A,vector)
    return r_spec



def deg_max(G,M): 
    vaccinated=[]
    # degree of all nodes sorted by index of the nodes
    degree_sequence = [val for (node, val) in sorted(G.degree(), key=lambda pair: pair[0])]
    degree_sequence2 = degree_sequence.copy()
    for i in range(M):
        maxi=max(degree_sequence2)
        vaccinated.append(degree_sequence.index(maxi))
        degree_sequence[degree_sequence.index(maxi)]=0
        degree_sequence2.remove(maxi)

    return vaccinated


# selection for vacc by max betweenness centrality with recomputation of centrality
def betw_max_(G,M): 

    G2 = rx.networkx_converter(G)
    vaccinated=[]
    # degree of all node sorted by index of the nodes
    dico=rx.betweenness_centrality(G2)
    centrality_sequence = list(dico.values())
    for i in tqdm(range(M)):
        maxi=max(centrality_sequence)
        node=[k for k in dico.keys() if dico[k] == maxi][0]
        vaccinated.append(node)
        G2.remove_node(node)
        dico=rx.betweenness_centrality(G2)
        centrality_sequence = list(dico.values())

    return vaccinated



# Configuration model with a gaussian distribution of degree
def config_model(N, med =10):
    std= 3 # standard deviation 
    distrib = np.random.normal(med,std,N).astype(int)
    if np.sum(distrib)%2 != 0:
        distrib[0]+=1
    G = nx.configuration_model(distrib)
    return G


# scale free graph with power law distribution 
def scale_free(N):
    G = nx.scale_free_graph(N)
    return G

def small_world(N):
    G=nx.newman_watts_strogatz_graph(N,10,0.4)
    return G
