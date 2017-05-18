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
        k = dictionary.keys()


        v = np.zeros(len(dictionary))

        # print v

        for term, index in dictionary.items():
            if term in vector:
                v[index] = 1

        #print dictionary[key]
        #print term1
        return v, dictionary, graph,k

    except Exception as e:
        print(str(e))

def build_matrix(dictionary, graph, v):
    length = len(dictionary)
    matrix = np.zeros((length, length))
    edge1=[]

    for edge in graph:
        edge1.append(edge)
        #print edge[0],dictionary[edge[0]]
        #print edge[1],dictionary[edge[1]]

        matrix[dictionary[edge[0]]][dictionary[edge[1]]] += 1
        # G.add_edge(edge[0], edge[1])
        #print matrix[dictionary[edge[0]]][dictionary[edge[1]]]
    #print matrix[137][137]

    #print edge1'''

    v = v / v.shape[0]

    return matrix, v,edge1

def distance(v1, v2):
    v = v1 - v2
    v = v * v
    return np.sum(v)

def compute(G, r0):
    "G is N*N matrix where if j links to i then G[i][j]==1, else G[i][j]==0"
    N = len(G)
    # print N
    d = np.zeros(N)
    # print d

    #Calculate the sum of columns, if 0 then make it as N
    for i in range(N):
        for j in range(N):
            if (G[j, i] > 0):
                d[i] += G[j, i]
                #print d[i]
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
    A = beta * M + T
    cnt=0
    while True:
        r1 = np.dot(A, r0)
        dist = distance(r1, r0)
        cnt +=1
        if dist < epsilon:
            break
        else:
            r0 = r1
    #print r1
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
        G.add_edge(edge[0], edge[1], weight=edge[2])

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

    #node_size = nx.get_edge_attributes(G, 'weight')
    n = [(d) for (u,v,d) in G.edges(data=True) if d >= 0.0]
    #node_size=node_size*1000
    j=[]
    k=[]
    for i in n:
        for key, value in i.iteritems():
            n=value*1000
            e=value*5
            j.append(n)
            k.append(e)
    #print j

    # draw graph
    nx.draw_networkx_nodes(G, graph_pos, node_size=j,
                           alpha=node_alpha, node_color=node_color)
    nx.draw_networkx_edges(G, graph_pos, width=k,
                           alpha=edge_alpha, edge_color=edge_color)
    nx.draw_networkx_labels(G, graph_pos, font_size=node_text_size,
                            font_family=text_font)

    labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, graph_pos, edge_labels=labels,
                                 label_pos=edge_text_pos)

    # show graph
    plt.show()

def Build_graph(dic):
    #print dic
    cnt=False
    for key1, value1 in dic.iteritems():
        for key2, value2 in value1.iteritems():
            graph = []
            for key3, value3 in value2.iteritems():
                if 'Chicago School' in key3:
                    val = 0
                    for key4, value4 in value3.iteritems():
                        edge = (key3, key4, value4)
                        graph.append(edge)
                        cnt=True
                        val=val+value4
                        print key1,key2,key4,value4
            if cnt == True:
                cnt=False
                print val,'\n'
                draw_graph(graph)


dict4 = {}  # doc year as key and id as value
for filename in glob.glob(os.path.join(NER_PATH, '*.txt')):
    #print filename
    i=1
    #if i==1:
    if filename == 'NER/2016_WritingTheStory.txt':
        dict3 = {}  # doc id as key and entities as value

        x = filename.rsplit('_', 1)[0][4:]  # year
        y = filename.rsplit('_', 1)[1].rsplit('.', 1)[0]  # doc id

        data = pickle.load(open(filename, 'rb'))
        # print data
        # process_content(data)
        v, dictionary, g, k = process_content(data)
        # print v
        M, V, E = build_matrix(dictionary, g, v)
        # print E

        R, c = compute(M, V)

        #print R

        dict1 = dict(zip(k, R.tolist()))
        p = int(round(len(dict1) * .3))
        # print p
        newDict1 = dict(sorted(dict1.iteritems(), key=operator.itemgetter(1), reverse=True)[:p])
        list4 = []

        m = {}
        dict2 = {}  # entities as key edges with weight as value

        for entities in newDict1.keys():
            dict1 = {}  # edges with weight

            temp_set = set()
            for edges in E:
                if entities == edges[0]:
                    # print entities,edges[1]
                    M[dictionary[edges[0]]][dictionary[edges[1]]]
                    dict1[edges[1]] = M[dictionary[edges[0]]][dictionary[edges[1]]]
                    # print entities,dict1
                    temp_set.add(edges[1])

                elif entities == edges[1]:
                    M[dictionary[edges[0]]][dictionary[edges[1]]]
                    dict1[edges[0]] = M[dictionary[edges[0]]][dictionary[edges[1]]]
                    temp_set.add(edges[0])
                    # print entities,dict1


                # print entities
                m[entities] = list(temp_set)
            # print entities,temp_set

            # print entities, dict1
            dict2[entities] = dict1
            dict3[y] = dict2
            # print entities
            # print dict1
    #print dict3
    dict4[x] = dict3

print dict4

#Build_graph(dict4)

'''# First, let's create a list of node sizes:
node_size = [data['number_of_edges'] for __, data in G.nodes(data=True)]
# Then, let's create a list of what the node colors should be:
#node_color = [
    #'blue' if data['gender'] == 'male' else 'red' for __, data in G.nodes(data=True)
]
# Finally, we pass this into the "draw_networkx" function:
nx.draw_networkx(
    G,
    pos=pos,
    node_size=node_size,
    node_color=node_color,
    edge_color='gray',
    alpha=0.3,
    font_size=14,
)
'''





