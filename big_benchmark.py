import networkx as nx
import retworkx as rx
import matplotlib.pyplot as plt
import random as rd
import numpy as np
from tqdm import tqdm
import sys 
from utils import *
sys.path.append('\gradient_methods')
import gradient_methods as gm
sys.path.remove('\gradient_methods')
sys.path.append('\other_methods')
import centrality as ce
import netshield as ns


def f_obj(config, Ao, eps =0.1, itemax =300):

    A = np.copy(Ao)
    l = config.astype(int)
    A = np.delete(A,l,0)
    A = np.delete(A,l,1)


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
    return r_spec



graph = ""

n_calc=10

G = nx.read_gml("dataset/"+"graph"+".gml")

N = G.number_of_nodes()
Gr = rx.networkx_converter(G)
A = rx.adjacency_matrix(Gr)


# compute the initial spectral radius
sr_init = 0
for i in range(n_calc):
    sr_init += max_ev(Gr)

sr_init /= n_calc


 ### Solution with grad downward
sol_grad_down = gm.gradient_downward(G,N)

sr_grad_down = [0]
l_k = [0]
l_index = list(range(N))
score1 = 0

for i in tqdm(range(0,N)):
    node = sol_grad_down[i]
    A = np.delete(A,l_index.index(node),0)
    A = np.delete(A,l_index.index(node),1)
    l_index.remove(node)
    eig=0
    for j in range(n_calc):
        eig += max_ev(A=A)
    sr_grad_down.append(sr_init-eig/n_calc)
    score1+=eig/n_calc
    l_k.append((i+1)/N)
print(score1)


### Solution with conj grad back
A= rx.adjacency_matrix(Gr)
sol_grad_up = gm.gradient_upward(G,N)

sr_grad_up = [0]
n_calc=20
A= rx.adjacency_matrix(Gr)
l_index = list(range(N))
score2 = 0

for i in tqdm(range(0,N)):
    node = sol_grad_up[i]
    A = np.delete(A,l_index.index(node),0)
    A = np.delete(A,l_index.index(node),1)
    l_index.remove(node)
    eig=0
    for j in range(n_calc):
        eig += max_ev(A=A)
    sr_grad_up.append(sr_init-eig/n_calc)
    score2+=eig/n_calc
print(score2)

np.save("sr_grad_down_and_up_"+graph+".npy",np.array([l_k,sr_grad_down,sr_grad_up]))

### solution with netshield 
A= rx.adjacency_matrix(Gr)

score3=0
sr_netsh = [0]

for k in tqdm(range(1,N)):
    vacc = ns.netshield(G,k,int(0.9+0.1*k))
    vacc = np.array(vacc)
    eig = 0
    for i in range(n_calc):
        eig += f_obj(vacc,A)
    sr_netsh.append(sr_init-eig/n_calc)
    score3+=eig/n_calc
print(score3)

np.save("sr_netshield_"+graph+".npy",sr_netsh)


### Solution with  degree centrality
A= rx.adjacency_matrix(Gr)
sol_deg=ce.deg_max(G,N)

sr_deg = [0]
score4 = 0
l_index = list(range(N))

for i in range(1,N+1):
    node = sol_deg[i]
    A = np.delete(A,l_index.index(node),0)
    A = np.delete(A,l_index.index(node),1)
    l_index.remove(node)
    eig=0
    for j in range(n_calc):
        eig += max_ev(A=A)
    score4+=eig/n_calc
    sr_deg.append(sr_init-eig/n_calc)
print(score4)


### Solution with betweenness centrality
A= rx.adjacency_matrix(Gr)
sol_betw=ce.betw_max(G,N)

sr_betw = [0]
score5 = 0
l_index = list(range(N))

for i in range(1,N+1):
    node = sol_betw[i]
    A = np.delete(A,l_index.index(node),0)
    A = np.delete(A,l_index.index(node),1)
    l_index.remove(node)
    eig=0
    for j in range(n_calc):
        eig += max_ev(A=A)
    score5+=eig/n_calc
    sr_betw.append(sr_init-eig/n_calc)
print(score5)


np.save("sr_centr_"+graph+".npy",np.array([sr_deg,sr_betw]))






plt.figure(1)
plt.plot(l_k,sr_grad_down/sr_init*100,label=" gradient downward")
plt.plot(l_k,sr_grad_up/sr_init*100,label="gradient upward")
plt.plot(l_k,sr_netsh/sr_init*100,label="netshield")
plt.plot(l_k,sr_deg/sr_init*100,label="degree centrality")
plt.plot(l_k,sr_betw/sr_init*100,label="betweenneess centrality")
plt.grid()
plt.xlabel('proportion of nodes immunized')
plt.ylabel("eigendrop (%)")
plt.legend()
plt.savefig("benchmark_"+graph+".png")
plt.show()

print('score gradient downward: ',score1)
print('score gradient upward: ',score2)
print('score netshield: ',score3)
print('score degree centrality: ',score4)
print('score betweenness centrality: ',score5)
