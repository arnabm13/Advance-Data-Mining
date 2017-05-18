import nltk
from nltk import pos_tag
from nltk.chunk import conlltags2tree
from nltk.tag import StanfordNERTagger
import sys
import re
import inspect
import os
import glob
import ntpath
import pickle
import re
import sc
import string

reload(sys)
sys.setdefaultencoding("ISO-8859-1")

SOURCE_PATH = 'output'
NER_PATH = 'NER'

def get_filename(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)

def stanfordNE2BIO(tagged_sent):
    bio_tagged_sent = []
    prev_tag = "O"
    for token, tag in tagged_sent:
        if tag == "O": #O
            if token != '':
                bio_tagged_sent.append((token, tag))
            prev_tag = tag
            continue
        if tag != "O" and prev_tag == "O": # Begin NE
            if token != '':
                bio_tagged_sent.append((token, "B-"+tag))
                prev_tag = tag
            else:
                prev_tag = "O"
        elif prev_tag != "O" and prev_tag == tag: # Inside NE
            if token != '':
                bio_tagged_sent.append((token, "I-"+tag))
                prev_tag = tag
            else:
                prev_tag = "O"
        elif prev_tag != "O" and prev_tag != tag: # Adjacent NE
            if token != '':
                bio_tagged_sent.append((token, "B-"+tag))
                prev_tag = tag
            else:
                prev_tag = "O"

    return bio_tagged_sent

def stanfordNE2tree(ne_tagged_sent):
    bio_tagged_sent = stanfordNE2BIO(ne_tagged_sent)
    sent_tokens, sent_ne_tags = zip(*bio_tagged_sent)
    sent_pos_tags = [pos for token, pos in pos_tag(sent_tokens)]

    sent_conlltags = [(token, pos, ne) for token, pos, ne in zip(sent_tokens, sent_pos_tags, sent_ne_tags)]
    ne_tree = conlltags2tree(sent_conlltags)
    return ne_tree

def extract_entity_names(t):
    entity_names = []
    if hasattr(t, 'label') and t.label:
        if t.label() in ['PERSON','FACILITY','ORGANIZATION','GSE','GSP','LOCATION']:
            entity = ' '.join([child[0] for child in t])
            entity_names.append((t.label(),entity))
        else:
            for child in t:
                entity_names.extend(extract_entity_names(child))

    return entity_names

def has_punctuation(word):
    regex = re.search(r'[\w./].*[\w./]', word)
    if regex:
        if regex.group(0) != word:
            return regex.group(0)
    return False

def has_intermediate_chars(word):
    g = re.search(r"^([^\.-]*)[\.-]*([^\.-]*)$",word)
    if g  and len(g.group(1)) != 0 and len(g.group(2)) != 0:
        return (g.group(1),g.group(2))
    else:
        return False

def has_compound_names(word):
    compound_words = re.findall('[a-zA-Z0-9][^A-Z]*', word)

    # Not split when we find the & between title. I.e. Barnes&Nobles
    regex = re.findall(r'[&][A-Z]*',word)
    if len(compound_words) > 1 and not word.isupper() and not regex:
        return compound_words
    else:
        return False

def get_parsed_word(word):
    result = []
    
    response_p = has_punctuation(word)
    response_ic = has_intermediate_chars(word)
    response_cn = has_compound_names(word)

    if not response_p and not response_ic and not response_cn:
        return sc.segment(word)

    # Remove punctuations at the beginning and end of the word. Except the '.', '-' and '/' character
    if response_p:
        # split the word into this punctuation and the word itself
        pre_charac = re.search(r'^([^\w./]*)', word)
        post_charac = re.search(r'([^\w./]*)$', word)
        if pre_charac:
            result.extend(get_parsed_word(pre_charac.group(0)))

        result.extend(get_parsed_word(response_p))

        if post_charac:
            result.extend(get_parsed_word(post_charac.group(0)))
    
    # Split a word if it has one or more dots or - in the middle of it. Example movie.theater into movie theater
    elif response_ic:
        result.extend(get_parsed_word(response_ic[0]))
        result.extend(get_parsed_word(response_ic[1]))
        
    # Split a word with a title format. I.e. ChicagoSchool into Chicago School, phraseChicago into phrase Chicago
    elif response_cn:
        for compound_word in response_cn:
            result.extend(get_parsed_word(compound_word))
    
    # Retrieve final result
    return result

    

def extract_ner(use_nltk = False):
    file_path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    st = StanfordNERTagger(file_path + os.sep + 'lib'+ os.sep + 'classifiers' +os.sep + 'english.all.3class.distsim.crf.ser.gz', file_path + os.sep + 'lib'+ os.sep + 'stanford-ner.jar')
    

    for filename in glob.glob(os.path.join(SOURCE_PATH,'*.txt')):
        print 'Processing file:',filename

        file = get_filename(filename)
        nerfname = os.path.join(NER_PATH,file)

        if not os.path.exists(nerfname):

            f = open(filename,'r')
            paragraphs = re.split('\.[^a-zA-Z]*\n',f.read())
            
            with open(nerfname,'wb') as outf:
            
                document_ner = []
                for i, paragraph in enumerate(paragraphs):
                    
                    sentences = nltk.sent_tokenize(paragraph)
                    tokenized_sentences = [nltk.word_tokenize(sentence) for sentence in sentences]
                    
                    # Manual cleaning of each of the words
                    stripped_sentences = []
                    for sentence in tokenized_sentences:
                        stripped_words = []
                        for word in sentence:
                            stripped_words.extend(get_parsed_word(word))

                        if stripped_words:
                            stripped_sentences.append(stripped_words)

                    # Name Entity Extraction Section
                    sentence_ner = []
                    if use_nltk:
                        # NLTK NER implementation
                        tagged_sentences = [nltk.pos_tag(sentence) for sentence in stripped_sentences]
                        tree = nltk.ne_chunk_sents(tagged_sentences)
                        sentence_ner = []
                        for stree in tree:
                            sentence_ner.extend(extract_entity_names(stree))
                    else:
                        # Standford NER implementation
                        for sentence in stripped_sentences:
                            chunks = st.tag(sentence)
                            tree = stanfordNE2tree(chunks)
                            sentence_ner.extend(extract_entity_names(tree))

                    # Add entities is list is not empty
                    if sentence_ner:
                        document_ner.append(sentence_ner)

                pickle.dump(document_ner,outf)
        break    

if not os.path.exists(NER_PATH):
    os.makedirs(NER_PATH)

extract_ner(use_nltk = False)
print 'Name Entity Extraction finished succesfully'
