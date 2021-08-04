import string
import numpy as np
import tensorflow as tf
from scipy import spatial
from nltk.tokenize import WordPunctTokenizer
from collections import Counter
from string import punctuation, ascii_lowercase
import regex as re
from tqdm import tqdm
import pandas as pd
from textblob import TextBlob
import spacy
from spacy import displacy
from collections import Counter
import en_core_web_sm
nlp = en_core_web_sm.load()
import nltk
import io
import time
spacy_nlp = spacy.load('en')
import re
import itertools
import xlsxwriter

def get_uncased_word_map(word_map):
    uncased_words = {}
    for key in word_map.keys():
        if key in uncased_words:
            uncased_words[key] = uncased_words[key] + word_map[key]
        else:
            uncased_words[key.lower()] = word_map[key]
    return uncased_words  

def get_word_list(word_map):
    return list(word_map.keys())

def save_dictionary_to_excel(save_dict, file_path):
    workbook = xlsxwriter.Workbook(file_path)
    workbook.encoding="utf-8"
    worksheet = workbook.add_worksheet()
    
    row = 0
    col = 0

    for key in save_dict.keys():
        worksheet.write(row, col, key)
        worksheet.write(row, col + 1, save_dict[key])
        row += 1

    workbook.close()

def isEnglish(s):
    try:
        s.encode(encoding='utf-8').decode('ascii')
    except UnicodeDecodeError:
        return False
    else:
        return True
    
def remove_non_latin_words(word_list):
    return list(filter(lambda t: isEnglish(t), word_list))    

def remove_non_latin_words_from_map(word_map):
    latin_words = {}
    for key in word_map.keys():
        if isEnglish(key):
            latin_words[key] = word_map[key]
    return latin_words  
    
    return map_book(list(filter(lambda t: isEnglish(t), get_word_list(word_map))))    

def remove_substring(substr, str):
    return re.sub(re.escape(substr), '', str)

def remove_punct(text):
    return ''.join((x for x in text if x not in string.punctuation))


def create_dict_embeddings(sentences, model, name):
    dict_embeddings = {}
    j = 1
    for i in range(len(sentences)):
        dict_embeddings[sentences[i]] = model.encode([sentences[i]], tokenize=True)
        if i >= (j * 1000) or i >= (len(sentences) - 1):
            d = {key:dict_embeddings[key] for i, key in enumerate(dict_embeddings)}
            file_name = "../data/squad/%s%s.npy" % (name,j)
            np.save(file_name, d) 
            j = j + 1
            print(j)
            dict_embeddings = {}
            
def load_embeddings(file_template, files_cnt, dict_emb):
    for i in range(files_cnt):
        print(i)
        file_name = "../data/squad/%s%s.npy" % (file_template,(i + 1))
        d = np.load(file_name).item()
        if len(dict_emb) == 0:
            dict_emb = dict(d)
        else :
            dict_emb.update(d)
    return dict_emb
            
def load_dict_embeddings(context_emb_cnt, questions_emb_cnt):
    dict_emb = {}
    dict_emb = load_embeddings('context_emb',context_emb_cnt,dict_emb)
    print('context loaded')
    dict_emb = load_embeddings('question_emb',questions_emb_cnt,dict_emb)
    print('questions loaded')
    return dict_emb

def create_dict_embeddings_no_save(sentences, model):
    j = 1
    dict_embeddings = {}
    print(j)
    for i in range(len(sentences)):
        if i >= (j * 1000):
            j = j + 1
            print(j)
        dict_embeddings[sentences[i]] = model.encode([sentences[i]], tokenize=True)
    return dict_embeddings
        
    
def get_target(x):
    idx = -1
    for i in range(len(x["sentences"])):
        if remove_punct(x["text"].lower()) in x["sentences"][i]: idx = i
    return idx

def predictions(train):
    
    train["cosine_sim"] = train.apply(cosine_sim, axis = 1)
    train["diff"] = (train["quest_emb"] - train["sent_emb"])**2
    train["euclidean_dis"] = train["diff"].apply(lambda x: list(np.sum(x, axis = 1)))
    del train["diff"]
    
    print("cosine start")
    
    train["pred_idx_cos"] = train["cosine_sim"].apply(lambda x: pred_idx(x))
    train["pred_idx_euc"] = train["euclidean_dis"].apply(lambda x: pred_idx(x))
    
    return train

def cosine_sim(x):
    li = []
    for item in x["sent_emb"]:
        li.append(spatial.distance.cosine(item,x["quest_emb"][0]))
    return li   

def pred_idx(distances):
    return np.argmin(distances)  


def pred_accuracy(target, predicted):
    
    acc = (target==predicted).sum()/len(target)
    
    return acc

def calc_tf_log_regression(train_y, train_x, test_y, test_x):
    num_features = train_x.shape[1]
    num_targets = 10

    train_y = np.column_stack(train_y)
    train_y = train_y.transpose()
    test_y = np.column_stack(test_y)
    test_y = test_y.transpose()

    trainy = np.empty((train_y.shape[0], 10))
    trainy.fill(0)
    for i in range(train_y.shape[0]):
        trainy[i][train_y[i]] = 1

    testy = np.empty((test_y.shape[0], 10))
    testy.fill(0)
    for i in range(test_y.shape[0]):
        testy[i][test_y[i]] = 1

    training_epochs = 40001
    tf.reset_default_graph()

    # By aving 2 features: hours slept & hours studied
    x = tf.placeholder(tf.float32, [None, num_features], name="X")
    y_ = tf.placeholder(tf.float32, [None, num_targets], name="Y")

    # Initialize our weigts & bias
    W = tf.get_variable("W", [num_features, num_targets], initializer = tf.contrib.layers.xavier_initializer())
    b = tf.get_variable("b", [num_targets], initializer = tf.zeros_initializer())

    y = tf.nn.softmax(tf.add(tf.matmul(x, W), b))

    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)

        for epoch in range(training_epochs):
            train_step.run({x: train_x, y_: trainy})   
            if (epoch + 1) % 1000 == 0:
                train_accuracy = accuracy.eval({x: train_x, y_: trainy})
                print('train acc %i : %f' % (epoch, train_accuracy))

        test_accuracy = accuracy.eval({x: test_x, y_: testy})
        print('test acc %f' % test_accuracy) 
        
def tokenize_sentence_uncased(text):
    non_letter_regex = '[\(\)\[\]:;–—/\\\\]+'
    #tokenize text to separate words
    separate_words = [word.strip(string.punctuation) for word in nltk.word_tokenize(text)]
    separate_words_regex = []
    for j in range(len(separate_words)):
        separate_words_regex += re.split(non_letter_regex, separate_words[j])
    #lowercase text
    separate_words_lowercased = [word.lower() for word in separate_words_regex if word]
    return separate_words_lowercased
        
def parse_sentence(text):
    document = spacy_nlp(text)
    sentence_without_NE = text
    named_entities = []
    non_letter_regex = '[\(\)\[\]:;–—/\\\\]+'
    # https://spacy.io/api/annotation
    NE_taggs = ['PERSON' , 'FAC', 'ORG', 'NORP', 'GPE', 'LOC', 'PRODUCT', 'LAW', 'LANGUAGE'] 
    for element in document.ents:
        if (element.label_ in NE_taggs):
            element_splitted = [word.strip(string.punctuation) for word in nltk.word_tokenize(element.text)]
            for j in range(len(element_splitted)):
                named_entities += re.split(non_letter_regex, element_splitted[j])
            #remove named entities from text
            sentence_without_NE = remove_substring(element.text, sentence_without_NE)
    #tokenize text without NE to separate words
    separate_words_without_NE = [word.strip(string.punctuation) for word in nltk.word_tokenize(sentence_without_NE)]
    separate_words_without_NE_regex = []
    for j in range(len(separate_words_without_NE)):
        separate_words_without_NE_regex += re.split(non_letter_regex, separate_words_without_NE[j])
    #lowercase text
    sentence_without_NE_lowercased = [word.lower() if word != 'I' else word for word in separate_words_without_NE_regex if word]
    return named_entities + sentence_without_NE_lowercased


def parse_sentences(list_sentences):
    words = []
    for sentence in tqdm(list_sentences):
        sentence_words = parse_sentence(sentence)
        words += sentence_words
        
    return remove_duplicates(words)


def map_book(tokens):
    hash_map = {}

    if tokens is not None:
        for element in tokens:
            # Remove Punctuation
            word = element.replace(",","")
            word = word.replace(".","")

            # Word Exist?
            if word in hash_map:
                hash_map[word] = hash_map[word] + 1
            else:
                hash_map[word] = 1

        return hash_map
    else:
        return None


#get all words from the sentence and their frequences
def get_words_and_freqs(list_sentences):
    words = []
    for sentence in tqdm(list_sentences):
        sentence_words = parse_sentence(sentence)
        words += sentence_words
    
    tokens_map = map_book(words)
    return tokens_map


# Convert one topic into two pandas tables. One table is for the context (context_table) 
# and another table stores the questions and answers (qas_table)
#
# The context table is defined as:  
#
# CREATE TABLE context_table (
#     context_id int,
#     context 'list of sentences'
# );
#
# The question table is defined as:
#
# CREATE TABLE qas_table (
#     context_id int,
#     questions sentence,
#     answers_start: int,
#     answers_text: sentence
# );
# The two tables can be joined by the context_id field.

def squad_to_table(topic):
    contexts = []
    context_id = []
    questions = []
    answers_text = []
    answers_start = []
    context_ref = []
    context_cnt = 0
    for section in topic:
        contexts.append(section['context'])
        context_id.append(context_cnt)
        for qas in section['qas']:
            if not qas['answers']:
                continue
            questions.append(qas['question'])
            answers_start.append(qas['answers'][0]['answer_start'])
            answers_text.append(qas['answers'][0]['text'])
            context_ref.append(context_cnt)
        context_cnt += 1
    
    context_table = pd.DataFrame({"context_id": context_id, "context":contexts})
    qas_table = pd.DataFrame({"context_id":context_ref, "questions": questions, 'answers_start':answers_start, 'answers_text':answers_text})
    return context_table, qas_table


alphabets= "([A-Za-z])"
prefixes = "(Mr|St|Mrs|Ms|Dr)[.]"
suffixes = "(Inc|Ltd|Jr|Sr|Co)"
starters = "(Mr|Mrs|Ms|Dr|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
websites = "[.](com|net|org|io|gov)"

def split_into_sentences(text):
    text = " " + text + "  "
    text = text.replace("\n"," ")
    text = re.sub(prefixes,"\\1<prd>",text)
    text = re.sub(websites,"<prd>\\1",text)
    if "Ph.D" in text: text = text.replace("Ph.D.","Ph<prd>D<prd>")
    text = re.sub("\s" + alphabets + "[.] "," \\1<prd> ",text)
    text = re.sub(acronyms+" "+starters,"\\1<stop> \\2",text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>\\3<prd>",text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>",text)
    text = re.sub(" "+suffixes+"[.] "+starters," \\1<stop> \\2",text)
    text = re.sub(" "+suffixes+"[.]"," \\1<prd>",text)
    text = re.sub(" " + alphabets + "[.]"," \\1<prd>",text)
    if "”" in text: text = text.replace(".”","”.")
    if "\"" in text: text = text.replace(".\"","\".")
    if "!" in text: text = text.replace("!\"","\"!")
    if "?" in text: text = text.replace("?\"","\"?")
    text = text.replace(".",".<stop>")
    text = text.replace("?","?<stop>")
    text = text.replace("!","!<stop>")
    text = text.replace("<prd>",".")
    sentences = text.split("<stop>")
    sentences = sentences[:-1]
    sentences = [s.strip() for s in sentences]
    return sentences

def get_squad_sentences(data_set):
    text = list()
    for topic_id in range(data_set.shape[0]):
        context_table, qas_table = squad_to_table(data_set.iloc[topic_id]['data']['paragraphs'])    
        text += list(context_table["context"].reset_index(drop= True))
        text += list(qas_table["questions"].reset_index(drop= True))
    return split_into_sentences(" ".join(text))

def remove_duplicates(words):
    non_empty_words = list(filter(None, words))
    return list(set(non_empty_words))

def remove_one_symbol_strings(l):
    strs = []
    for t in l:
        if len(remove_punct(t)) > 1 :
            strs.append(t)  
    return strs

def loadGloveModel(gloveFile):
    print ("Loading Glove Model")
    model = {}
    with io.open(gloveFile, encoding='utf-8') as f:
        for line in f:
            splitLine = line.split()
            word = splitLine[0]
            embedding = np.array([float(val) for val in splitLine[1:]])
            model[word] = embedding
    print ("Done.",len(model)," words loaded!")
    return model

def get_missing_words(name, model, words):
    start_time = time.time()
    words_out_of_model = []
    word_range = tqdm(words)
    for word in word_range:
        if not(word in model):
            words_out_of_model.append(word)
    word_range.close()
    print('words out of {0} : {1}, percentage of successful word embedding {2}%, unsuccessful {3}%'.format(name, len(words_out_of_model), 
                                                                 100 - len(words_out_of_model) / (len(words)) * 100,
                                                                 len(words_out_of_model) / (len(words)) * 100))
    print("Done. Time: {} sec.".format(round(time.time()-start_time, 2)))
    return words_out_of_model

def get_word_existence_for_WE(model, words):
    word_existence = []
    word_range = tqdm(words)
    for word in word_range:
        if not(word in model):
            word_existence.append(0)
        else:
            word_existence.append(1)
    return word_existence

def process_test_words(model_names, model_case, test_words, test_words_uncased, models):
    test_df = pd.DataFrame(columns= ['name'] + test_words)
    for i in range(len(models)):
        test_df.loc[i,'name'] = model_names[i]
        test_words_curr = []
        if model_case[i] == 'cased':
            test_words_curr = test_words
        else:
            test_words_curr = test_words_uncased
        for j in range(len(test_words_curr)):
            if not(test_words_curr[j] in models[i]):
                test_df.loc[i,test_words[j]] = 'not in dict'
            else:
                test_df.loc[i,test_words[j]] = 'in dict'
    return test_df

def get_accuracy(positives, reference_positives, total_words_count, df, df_number, name):
    positives_splitted =  list(itertools.chain.from_iterable([x.split() for x in positives]))
    false_positive_count = 0
    true_positive_count = 0
    positive_count = len(reference_positives)
    for i in range(len(positives_splitted)):
        if positives_splitted[i] not in reference_positives:
            false_positive_count += 1
        else:
            true_positive_count += 1
    FPR = false_positive_count / (total_words_count - positive_count)
    TPR = true_positive_count / positive_count
    FNR = 1-TPR
    Precision = true_positive_count / len(positives_splitted)
    print('FPR: %f' % FPR)
    print('Recall(TPR): %f' % TPR)
    print('FNR: %f' % FNR)
    print('Precision: %f' % Precision)
    df.loc[df_number] = [name, FPR, TPR, FNR, Precision]
       
def f1_score(prediction, ground_truth):
    prediction_tokens = parse_sentence(prediction)
    ground_truth_tokens = parse_sentence(ground_truth)
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction, ground_truth):
    return (parse_sentence(prediction) == parse_sentence(ground_truth))