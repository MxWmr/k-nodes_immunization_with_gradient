import numpy as np
import retworkx as rx 
from tqdm import tqdm


def power_iteration(B,eps =0.0001, itemax =10000):
    

    b_k = np.random.rand(B.shape[0])

    b_k1_norm = np.linalg.norm(b_k)
    v=0
    ite=0
    while abs(v-b_k1_norm)>eps and ite<itemax:
        v = b_k1_norm

        # calculate the matrix-by-vector product Ab
        b_k1 = B.dot(b_k)


        # calculate the norm
        b_k1_norm = np.linalg.norm(b_k1,ord=2)

        # re normalize the vector

        b_k = 1/(b_k1_norm+0.001)*b_k1

        ite+=1

    return b_k,np.where(b_k!=0.,1/b_k/1000,0)



def grad_comput(A,eta):
    B = np.dot(A,np.diag(eta))
    psi,phi = power_iteration(B)
    return np.dot(np.diag(phi),np.dot(A.T,psi))


def  gradient_downward(G,N):
    """
    in:
    G: graph to immunize
    N: number of nodes

    out:
    vaccinated: a list with all node indices ordered by their vaccination 
    """

    G_r = rx.networkx_converter(G)
    vaccinated = []
    A = rx.adjacency_matrix(G_r)
    eta = np.ones([N])

    for n in tqdm(range(N)):

        grad = grad_comput(A,eta)

        try:
            node = np.argmin(grad)[0]
        except:
            node = np.argmin(grad)
        grad[node] = np.Inf

        while eta[node] == 0:
            try:
                node = np.argmin(grad)[0]
            except:
                node = np.argmin(grad)
            grad[node] = np.Inf

        eta[node] = 0

        vaccinated.append(node)

    return vaccinated



def gradient_upward(G,N):
    """
    in:
    G: graph to immunize
    N: number of nodes

    out:
    vaccinated: a list with all node indices ordered by their vaccination 
    """

    G_r = rx.networkx_converter(G)
    vaccinated = []
    A = rx.adjacency_matrix(G_r)
    eta = np.zeros([N])

    for n in tqdm(range(N)):

        grad = grad_comput(A,eta)
        try:
            node = np.argmax(grad)[0]
        except:
            node = np.argmax(grad)
        grad[node] = np.NINF

        while eta[node] == 1:
            try:
                node = np.argmax(grad)[0]
            except:
                node = np.argmax(grad)
            grad[node] = np.NINF

        eta[node] = 1

        vaccinated.append(node)

    vaccinated.reverse()
    return vaccinated


