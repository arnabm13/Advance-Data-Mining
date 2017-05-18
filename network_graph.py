import nltk
from nltk.tokenize import PunktSentenceTokenizer
import re
from collections import defaultdict
import glob
import os
import sys
import numpy as np
import pickle
import operator
import networkx as nx
import matplotlib.pyplot as plt

"Power iteration with random teleports that addresses Spider trap problem or Dead end problem "
beta = 0.85
epsilon = 0.0001
NER_PATH = 'NER'
reload(sys)
sys.setdefaultencoding("ISO-8859-1")

def get_dictionary(nodes):
    response = {}
    i = 0
    for key, value in nodes.items():
        response[key] = i
        i += 1
    return response

def process_content(tokenized):
    try:
        #print tokenized
        vector = []
        nodes = {}
        graph = []

        for i in tokenized[:]:
            #print i[0]
            new=True
            for j in i[:]:
                j = j[1]
                if new:
                    vector.append(j)
                    new=False
                if j in nodes:
                    val = nodes[j]
                    nodes[j] = val + 1
                    edge = (temp, j)
                    graph.append(edge)
                    temp = j

                else:
                    if len(nodes) == 0:
                        temp = j
                        nodes[j] = 1
                    else:
                        nodes[j] = 1
                        edge = (temp, j)
                        graph.append(edge)
                        temp = j
        #print nodes
        #print graph
        #print vector
        dictionary = get_dictionary(nodes)
        #print dictionary

        v = np.zeros(len(dictionary))
        #print v
        for term, index in dictionary.items():
            if term in vector:
                v[index] = 1
        #print v
        return v, dictionary, graph

    except Exception as e:
        print(str(e))

def build_matrix(dictionary, graph, v):
    length = len(dictionary)
    matrix = np.zeros((length, length))

    for edge in graph:
        # print edge[0],',',edge[1]
        matrix[dictionary[edge[0]]][dictionary[edge[1]]] += 1
        # G.add_edge(edge[0], edge[1])
        #print matrix[dictionary[edge[0]]][dictionary[edge[1]]]
    #print matrix

    v = v / v.shape[0]
    return matrix, v

def distance(v1, v2):
    v = v1 - v2
    v = v * v
    return np.sum(v)

def compute(G, r0):
    "G is N*N matrix where if j links to i then G[i][j]==1, else G[i][j]==0"
    N = len(G)
    #print G
    d = np.zeros(N)
    #print d

    #Calculate the sum of columns, if 0 then make it as N
    for i in range(N):
        for j in range(N):
            if (G[j, i] > 0):
                d[i] += G[j, i]
                #print d
        #print d
        if d[i]==0:   # i is dead end, teleport always
            d[i] = N
    #print d

    #Initial vector
    r0 = np.zeros(N, dtype=np.float32) + 1.0 / N
    #print r0

    # construct stochastic M (Normalise matrix M)
    M = np.zeros((N, N), dtype=np.float32)
    #print M
    for i in range(N):
        if (d[i]==N):  # i is dead end
            for j in range(N):
                M[j, i] = 1.0 / d[i]
        else:
            for j in range(N):
                if G[j, i] > 0:
                    M[j, i] = G[j, i] / d[i]
        #print M

    #Page rank Equation
    T = (1.0 - beta) * (1.0 / N) * (np.zeros((N, N), dtype=np.float32) + 1.0)
    #print beta
    A = beta * M +  T
    cnt=0
    while True:
        r1 = np.dot(A, r0)
        dist = distance(r1, r0)
        cnt +=1
        if dist < epsilon:
            break
        else:
            r0 = r1

    return r1,cnt

def draw_graph(graph, labels=None, graph_layout='shell',
               node_size=1600, node_color='blue', node_alpha=0.3,
               node_text_size=12,
               edge_color='blue', edge_alpha=0.3, edge_tickness=1,
               edge_text_pos=0.3,
               text_font='sans-serif'):


    # create networkx graph
    G = nx.Graph()

    # add edges
    for edge in graph:
        G.add_edge(edge[0], edge[1])

    # these are different layouts for the network you may try
    # shell seems to work best
    if graph_layout == 'spring':
        graph_pos = nx.spring_layout(G)
    elif graph_layout == 'spectral':
        graph_pos = nx.spectral_layout(G)
    elif graph_layout == 'random':
        graph_pos = nx.random_layout(G)
    else:
        graph_pos = nx.shell_layout(G)


    # draw graph
    nx.draw_networkx_nodes(G, graph_pos, node_size=node_size,
                           alpha=node_alpha, node_color=node_color)
    nx.draw_networkx_edges(G, graph_pos, width=edge_tickness,
                           alpha=edge_alpha, edge_color=edge_color)
    nx.draw_networkx_labels(G, graph_pos, font_size=node_text_size,
                            font_family=text_font)


    # show graph
    plt.show()

for filename in glob.glob(os.path.join(NER_PATH,'*.txt')):
    print filename
    data = pickle.load(open(filename, 'rb'))
    print data
    #process_content(data)
    v,dictionary,g=process_content(data)
    draw_graph(g)

    #print v
    #M,V = build_matrix(dictionary, g, v)
    #print M

    #R,c= compute(M, V)
    #print c
    #print R
