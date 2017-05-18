import pickle
import os
import glob
import ntpath
import sys

reload(sys)
sys.setdefaultencoding("ISO-8859-1")

NER_PATH = 'NER'
CROSSED_PATH = 'CROSSED_DATA'

if not os.path.exists(CROSSED_PATH):
    os.makedirs(CROSSED_PATH)

def get_filename(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)

def read_file(filename):
    names = []
    with open(filename, 'r') as f:
        for line in f:
            if line.strip() != '':
                names.append(line.strip())
    return names

def find_in_names(entity, bd_architect_names, bd_companies_names):
    for bd_name in bd_companies_names:
        if bd_name in entity:
            #print 'Entity:',entity,' Company Name:',bd_name
            return bd_name
    for bd_name in bd_architect_names:
        if bd_name in entity:
            #print 'Entity:',entity,' Architect Name:',bd_name
            return bd_name

    #print 'Removed:',entity
    return ''

if not os.path.exists(NER_PATH):
    raise 'NER Directory with entities must be created first. Run the script ner.py before'

architects_names = read_file('data/architects_names.txt')
companies_names = read_file('data/companies_names.txt')
response = []

for filename in glob.glob(os.path.join(NER_PATH,'*.txt')):
    file = get_filename(filename)
    
    lines = pickle.load(open(filename,'rb'))
    response = []
    for paragraph in lines:
        
        response_parag = []
        for entity in paragraph:
        
            my_entity = find_in_names(entity[1],architects_names,companies_names)
        
            if my_entity != '':
                response_parag.append((entity[0],my_entity))
        
        if response_parag:
            response.append(response_parag)

    if response:
        with open(os.path.join(CROSSED_PATH,file),'wb') as f:
            pickle.dump(response, f)
