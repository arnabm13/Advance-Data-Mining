import os 
import glob
import pickle
import sys
import re
from collections import Counter, defaultdict, OrderedDict
from operator import itemgetter

reload(sys)
sys.setdefaultencoding("ISO-8859-1")

INPUT_PATH = 'NER'
N_GRAM = 1
R_FREQ = 1
QUERIES = [('Chicago','Chicago'),('Chicago','Chicago'),('Chicago','Chicago')]


def get_data_from_title(title):
    values = title.split(os.sep)
    my_data = values[1].split('_')
    if len(my_data) != 2:
        print len(my_data)
        raise Exception('Error n file ',title,'. It requires to have the YEAR and ID in the name of the file')
    return my_data[0], re.sub(r'.txt','',my_data[1])

def get_removed_terms(data, rfreq):
    c = Counter()
    for paragraph in data:
        c.update(elem[1] for elem in paragraph)
    return [k for k,v in c.iteritems() if v <= rfreq]

def get_top(data):
    response = defaultdict(Counter)
    for dictionary in data:
        response[dictionary['year']].update(dictionary['n_grams'])

    response = OrderedDict(sorted(response.items()))
    r_filename = os.path.join('result.txt')   
    with open(r_filename,'wb') as f: 
        for year,n_grams in response.items():
            string = str(year) +'\t\t'
            for ngram in n_grams.most_common()[:10]:
                string += ngram[0][0] + ':' + str(ngram[1]) + '\t'
            f.write(string + '\n\n')


def get_n_grams(data, ngram, forbidden_list):
    new_data = []
    for paragraph in data:
        new_paragraph = []
        for type_ner, term in paragraph:
            if term not in forbidden_list:
                new_paragraph.append(term)
        if new_paragraph:
            new_data.append(new_paragraph)
    
    ngram_list = []
    for paragraph in new_data:
        if len(paragraph) >= ngram:
            
            for i in range(len(paragraph) - (ngram - 1)):
                n_tuple = []
                for j in range(ngram):
                    n_tuple.append(paragraph[i+j].upper())
                ngram_list.append(tuple(n_tuple))
    
    return Counter(ngram_list)

def query(data):
    response = defaultdict(float)
    for dictionary in data:
        for query in QUERIES:
            if dictionary['n_grams'][query]:
                response[dictionary['year']] += dictionary['n_grams'][query]

    response2 = defaultdict(list)
    for dictionary in data:
        for query in QUERIES:
            if dictionary['n_grams'][query]:
                response2[dictionary['year']].append(dictionary['id'])
    return OrderedDict(sorted(response.items())), OrderedDict(sorted(response2.items())) 


def get_tree_values(ngram, rfreq, retrieve_all = True):
    all_data = []
    for filename in glob.glob(os.path.join(INPUT_PATH,'*.txt')):
        doc_info = {}
        year, id = get_data_from_title(filename)
        doc_info['year'] = year
        doc_info['id'] = id
        
        data = pickle.load(open(filename,'rb'))
        forbidden_list = get_removed_terms(data,rfreq)
        doc_info['n_grams'] = get_n_grams(data, ngram, forbidden_list)

        all_data.append(doc_info)

    if retrieve_all:
        get_top(all_data)
    else:
        response, response2 = query(all_data)
        print response,'\n'
        print response2
             

if __name__ == "__main__":
    get_tree_values(N_GRAM, R_FREQ, True)
    