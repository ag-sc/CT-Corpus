
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.layers import Embedding,Conv1D,Dense,Dropout,LSTM,Bidirectional,TimeDistributed
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.initializers import Constant

import numpy as np
import random
import itertools
import copy
import sys
import pickle
from ctro import *

# settings #####################################################################
BATCH_SIZE = 10
MAX_SENTENCE_LENGTH = 100
MIN_LABEL_FREQ = 20

SETTING_WORD_VECTORS = 'wikipedia'
#SETTING_WORD_VECTORS = 'pmc'

SETTING_DISEASE = sys.argv[1]
if SETTING_DISEASE != 'gl' and SETTING_DISEASE != 'dm2':
    print('Invalid disease parameter!')
    sys.exit()

# path ########################################################################

if SETTING_WORD_VECTORS == 'wikipedia':
    WORD_VECTORS_PATH = '/homes/cwitte/word_vectors/glove_wikipedia/glove.6B.100d.txt'
else:
    WORD_VECTORS_PATH = '/homes/cwitte/word_vectors/pmc_pubmed_wikipedia/word_vectors.txt'

if SETTING_DISEASE == 'gl':
    DATA_DUMP_PATH = 'glaucoma_slot_filling.dump'
elif SETTING_DISEASE == 'dm2':
    DATA_DUMP_PATH = 'diabetes_slot_filling.dump'
    
#DATA_DUMP_PATH = '/home/cwitte/annotation/' + DATA_DUMP_PATH
    
    
def annotation_lists_equality(l1, l2):
    for annotation1 in l1:
        found = False
        
        for annotation2 in l2:
            if annotation1.left_context == annotation2.left_context:
                if annotation1.tokens == annotation2.tokens:
                    if annotation1.right_context == annotation2.right_context:
                        if annotation1.most_general_superclass == annotation2.most_general_superclass:
                            found = True
                            break
                
        if not found:
            return False
            
    return True



def convert_sentence_taggings_to_annotations(sentence_taggings, sentence_tokens, sentence_numbers):
    assert len(sentence_taggings) == len(sentence_tokens)
    
    annotations = []
    current_sentence_offset = 0
    
    for i in range(len(sentence_taggings)):
        tagging = sentence_taggings[i]
        tokens = sentence_tokens[i]
        
        for offset in tagging.taggings:
            for most_general_superclass,length in tagging.taggings[offset]:
                annotation = Annotation()
                
                annotation.left_context = tokens[:offset]
                annotation.tokens = tokens[offset:offset+length]
                annotation.right_context = tokens[offset+length:]
                annotation.most_general_superclass = most_general_superclass
                annotation.doc_char_onset = current_sentence_offset + len(annotation.left_context)
                annotation.sentence_number = sentence_numbers[i]
            
                annotations.append(annotation)
                
        current_sentence_offset += len(tokens)
                
    return annotations
                


def compute_micro_stats(stats_dict):
    stat_result = F1Statistics()
    
    for class_name in stats_dict:
        stat_result.true_positives += stats_dict[class_name].true_positives
        stat_result.false_positives += stats_dict[class_name].false_positives
        stat_result.num_occurences += stats_dict[class_name].num_occurences
        
    return stat_result


def print_stats(stats_dict):
    for class_name in stats_dict:
        precision = stats_dict[class_name].precision()
        recall = stats_dict[class_name].recall()
        f1 = stats_dict[class_name].f1()
        print(class_name + ':')
        print('precision:' + str(precision) + '; recall:' + str(recall) + '; f1:' + str(f1))
        
    micro_stat = compute_micro_stats(stats_dict)
    precision = micro_stat.precision()
    recall = micro_stat.recall()
    f1 = micro_stat.f1()
    print('micro:')
    print('precision:' + str(precision) + '; recall:' + str(recall) + '; f1:' + str(f1))
    


def print_abstract_sentence(abstract, query_sentence_number):
    tokens = []
    labels = {}
    for i in range(MAX_SENTENCE_LENGTH):
        labels[i] = []
    
    for (sentence_number, doc_char_onset, token) in abstract.tokenization.tokens:
            if sentence_number == query_sentence_number:
                tokens.append(token)
                
    for annotation in abstract.annotated_abstract.annotations:
        if annotation.sentence_number != query_sentence_number:
            continue
        if annotation.most_general_superclass not in ontology.used_most_general_classes:
            continue
        
        offset = len(annotation.left_context)
        length =  len(annotation.tokens)
        for i in range(length):
            labels[offset+i].append(annotation.most_general_superclass)
        
    for i,token in enumerate(tokens):
        print(token + ':' + str(labels[i]))



class SentenceTagging:
    def __init__(self):
        # offset -> set((label,length))
        self.taggings = {}
        
        
    def set_from_abstract(self, abstract, sentence_number):
        annotated_abstract = abstract.annotated_abstract
        self.taggings = {}
        
        for annotation in annotated_abstract.annotations:
            if annotation.sentence_number != sentence_number:
                continue
            if annotation.most_general_superclass not in ontology.used_most_general_classes:
                continue
            
            offset = len(annotation.left_context)
            length = len(annotation.tokens)
            label  = annotation.most_general_superclass
            
            # add tagging
            if offset not in self.taggings:
                self.taggings[offset] = set()

            self.taggings[offset].add((label,length))
                
    
    # convert to (sentence_lenghth,num_iob_labels) binary numpy array
    def encode(self, sentence_length, label_indices):
        matrix = np.zeros((sentence_length, len(label_indices)))
        
        for offset in self.taggings:
            for label,length in self.taggings[offset]:
                
                # start pos of annotation
                label_iob = label + '-B'
                if label_iob not in label_indices:
                    continue
                
                label_index = label_indices[label_iob]
                matrix[offset,label_index] = 1
                
                # interior positions of annotation
                if length > 1:
                    label_index = label_indices[label + '-I']
                    
                    for j in range(1,length):
                        matrix[offset+j,label_index] = 1
                        
        # add O label for tokens having no annotation
        mat_reduced = np.sum(matrix, axis=-1)
        
        for i in range(sentence_length):
            if mat_reduced[i] == 0:
                matrix[i, label_indices['O']] = 1
                        
        return matrix
    
    
    # convert from (sentence_lenghth,num_iob_labels) binary numpy
    def set_fom_encoding(self, matrix, labels_original, label_indices):
        self.taggings = {}
        sentence_length = np.shape(matrix)[0]
        
        for i in range(sentence_length):
            for label in labels_original:
                label_index = label_indices[label + '-B']
                
                if matrix[i,label_index] == 1:
                    offset = i
                    length = 1
                    
                    for j in range(i+1, sentence_length):
                        label_index = label_indices[label + '-I']
                        
                        if matrix[j,label_index] == 1:
                            length += 1
                        else:
                            break
                    
                    # add tagging
                    if offset not in self.taggings:
                        self.taggings[offset] = set()
                        
                    self.taggings[offset].add((label,length))
                    
                    
    def to_set(self, query_label, b_exact=True):
        taggings_set = set()
        
        for offset in self.taggings:
            for label,length in self.taggings[offset]:
                if label != query_label:
                    continue

                if b_exact:
                    taggings_set.add((offset,label,length))
                else:
                    taggings_set.add((offset,label))
                    
        return taggings_set
        
    
        
class Sentence:
    def __init__(self):
        self.tokens = []
        self.abstract_id = None
        self.sentence_number = None
        self.sentence_tagging = None
        
    def set_from_abstract(self, abstract, sentence_number):
        self.tokens = []
        self.abstract_id = abstract.abstract_id
        self.sentence_number = sentence_number
        self.sentence_tagging = SentenceTagging()
        
        # estimate tokens of sentence
        for (sentence_number, doc_char_onset, token) in abstract.tokenization.tokens:
            if sentence_number == self.sentence_number:
                self.tokens.append(token)
                
        # estimate taggings
        self.sentence_tagging.set_from_abstract(abstract, self.sentence_number)
        
        
    def print_out(self, indices_iob_labels, indices_iob_labels_reverse):
        tagging_encoded = self.sentence_tagging.encode(MAX_SENTENCE_LENGTH, indices_iob_labels)
        
        for i in range(len(self.tokens)):
            labels = []
            
            for label_index in indices_iob_labels_reverse:
                if tagging_encoded[i,label_index] == 1:
                    labels.append(indices_iob_labels_reverse[label_index])
                    
            print(self.tokens[i] + ':' + str(labels))
            
            
    def to_annotations(self):
        annotations = []
        
        
        
class TaggingBasline:
    def __init__(self):
        self.labels_original = None

        self.indices_labels_iob = None
        self.indices_labels_iob_reverse = None
        
        self.abstracts_train = None
        self.abstracts_validation = None
        self.abstracts_test = None
        
        # [abstract_indices}[sentence_indices]]
        self.sentences_train = None
        self.sentences_validation = None
        self.sentences_test = None
        
        self.feed_dicts_train = None
        self.feed_dicts_validation = None
        self.feed_dicts_test = None
        
        self.session = None
        
        
        
    def import_abstracts(self):
        f = open(DATA_DUMP_PATH, 'rb')
        self.abstracts_train,self.abstracts_validation,self.abstracts_test = pickle.load(f, encoding='latin1')
        f.close()
        
        for abstract in self.abstracts_train:
            abstract.remove_inner_annotations()
            abstract.annotated_abstract.assign_new_global_indices()
            
        for abstract in self.abstracts_validation:
            abstract.remove_inner_annotations()
            abstract.annotated_abstract.assign_new_global_indices()
            
        for abstract in self.abstracts_test:
            abstract.remove_inner_annotations()
            abstract.annotated_abstract.assign_new_global_indices()
     
        
        
    def extract_sentences_from_abtract(self, abstract):
        sentence_numbers = abstract.get_sentence_numbers()
        sentences = []
        
        for sentence_number in sentence_numbers:
            sentence = Sentence()
            sentence.set_from_abstract(abstract, sentence_number)
            sentences.append(sentence)
            
        return sentences
    
    
    '''   
    def create_feed_dict_from_sentences(self, sentences):
        # list of numpy array containing encoded tokens for
        # sentence of batch
        encoded_tokens = []
            
        # list of numpy array containing encoded lables for each
        # sentence of batch
        encoded_labels = []
        
        for sentence in sentences:
            # encode tokens
            encoded_tokens_sentence = np.zeros((MAX_SENTENCE_LENGTH,))
            for i,token in enumerate(sentence.tokens):
                token = self.encode_token(token)
                encoded_tokens_sentence[i] = token
                
            encoded_tokens.append(encoded_tokens_sentence)
            encoded_labels.append(sentence.sentence_tagging.encode(MAX_SENTENCE_LENGTH, self.indices_labels_iob))
            
        feed_dict = {}
        feed_dict[self.placeholders_tokens] = np.stack(encoded_tokens, axis=0)
        feed_dict[self.placeholders_labels] = np.stack(encoded_labels, axis=0)
        return feed_dict
    '''    


    def create_graph(self):
        self.placeholder_tokens = tf.placeholder(shape=(None,None), dtype=tf.float32)
        self.placeholder_labels = tf.placeholder(shape=(None,None), dtype=tf.float32)
        
        embedding_layer = Embedding(self.vocab_size+3, self.embedding_dim, trainable=False)
        embedding_layer.build(input_shape=(1,))
        embedding_layer.set_weights([self.embedding_matrix])
        
        linear_layer = Dense(len(self.indices_labels_iob))
        
        x = embedding_layer(self.placeholder_tokens)
        x = Bidirectional(LSTM(units=100, return_sequences=True))(x)
        logits = TimeDistributed(linear_layer)(x)
        logits = tf.squeeze(logits, axis=[0])

        loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.placeholder_labels, logits=logits)
        loss = tf.reduce_sum(loss)
        self.optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(loss)
        
        # prediction
        taggings = tf.greater(logits, 0)
        taggings = tf.cast(taggings, tf.int32)
        self.taggings = taggings
    
    
    
    def create_feed_dicts_from_abstracts(self, abstracts):
        sentences = []
        feed_dicts = []
        
        # extract sentences from abstracts
        for abstract in abstracts:
            new_sentences = self.extract_sentences_from_abtract(abstract)
            sentences.extend(new_sentences)
        
        # create feed dicts     
        for sentence in sentences:
            feed_dict = {}
            tokens = [self.encode_token(token) for token in sentence.tokens]
            
            feed_dict[self.placeholder_tokens] = [tokens]
            feed_dict[self.placeholder_labels] = sentence.sentence_tagging.encode(len(tokens), self.indices_labels_iob)

            feed_dicts.append(feed_dict)
            
        return sentences, feed_dicts
    
    
    
    def create_feed_dicts(self):
        self.sentences_train, self.feed_dicts_train = self.create_feed_dicts_from_abstracts(self.abstracts_train)
        self.sentences_validation, self.feed_dicts_validation = self.create_feed_dicts_from_abstracts(self.abstracts_validation)
        self.sentences_test, self.feed_dicts_test = self.create_feed_dicts_from_abstracts(self.abstracts_test)

        
        
    def import_word_vectors(self):
        # import word vectors ###############################################
        word_vectors_dict = {}
        f = open(WORD_VECTORS_PATH)
        
        parts_prev = None
    
        if 'pubmed' in WORD_VECTORS_PATH:
            f.readline()
        
        for line in f:
            parts = line.strip().split()
            
            if parts_prev != None  and  len(parts_prev) != len(parts):
                continue
                
            token = parts[0]
            l = [float(s) for s in parts[1:]]
            vector = np.array(l, dtype=np.float32)
            word_vectors_dict[token] = vector
            parts_prev = parts
            
        # compute average embedding #########################################################
        dim = len(word_vectors_dict['the'])
        avg_word_vector = np.zeros(dim, dtype=np.float32)
        
        for token,vec in word_vectors_dict.items():
            if dim != len(vec):
                print(token)
            avg_word_vector += vec
    
        avg_word_vector /= len(word_vectors_dict)
        
        # compute word vector indices #########################################################
        embedding_dim = len(avg_word_vector)
        vocab_size = len(word_vectors_dict)
        word_vector_indices = {}
        embedding_matrix = np.zeros((vocab_size+3, embedding_dim)) # + 3 beacause of UNK, NONE, 0 masking
        
        for i,token in enumerate(list(word_vectors_dict.keys())):
            # i+1 because index 0 is reserved for marking
            word_vector_indices[token] = i+1
            embedding_matrix[i+1] = word_vectors_dict[token]
        
        # set index of special word vectors
        unk_vector_index = vocab_size + 1 # + 1 because index 0 is reserved for masking
        none_vector_index = vocab_size + 2
        
        # add UNK vector to embedding matrix
        embedding_matrix[unk_vector_index] = avg_word_vector
        assert len(word_vectors_dict) == len(word_vector_indices)
        
        # create reverse mapping for word vector indices
        reverse_word_vector_indices = {}
        for token,index in word_vector_indices.items():
            reverse_word_vector_indices[index] = token
            
        reverse_word_vector_indices[0] = '0'
        reverse_word_vector_indices[unk_vector_index] = 'UNK'
        reverse_word_vector_indices[none_vector_index] = 'NONE'
        
        word_vector_indices['UNK'] = unk_vector_index
        word_vector_indices['NONE'] = none_vector_index
        
        # save data    
        self.word_vectors_indices = word_vector_indices
        self.reverse_word_vector_indices = reverse_word_vector_indices
        self.embedding_matrix = embedding_matrix
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        
        
        
    # token -> index word_embedding_matrix
    def encode_token(self, token):
        if token in self.word_vectors_indices:
            index = self.word_vectors_indices[token]
        else:
            index = self.word_vectors_indices['UNK']
        
        return index
    
    
    
    def print_label_frequency(self, abstracts):
        labels_count = {}
        
        # extract original lables
        for abstract in abstracts:
            annotated_abstract = abstract.annotated_abstract

            for annotation in annotated_abstract.annotations:
                label = annotation.most_general_superclass
                
                if label not in ontology.used_most_general_classes:
                    continue
                if label not in labels_count:
                    labels_count[label] = 1
                else:
                    labels_count[label] += 1
                    
        for label in sorted(labels_count.keys()):
            print(label + ':' + str(labels_count[label]))
                    
    
        
    def extract_labels(self, abstracts):
        self.labels_original = set()
        labels_count = {}
        
        # extract original lables
        for abstract in abstracts:
            annotated_abstract = abstract.annotated_abstract

            for annotation in annotated_abstract.annotations:
                label = annotation.most_general_superclass
                
                if label not in ontology.used_most_general_classes:
                    continue
                if label not in labels_count:
                    labels_count[label] = 1
                else:
                    labels_count[label] += 1
                    
        # only keep labels with min freq
        for label in labels_count:
            if labels_count[label] >= MIN_LABEL_FREQ:
                self.labels_original.add(label)
                
            print(label + ': ' + str(labels_count[label]))
                
        # create iob labels
        self.indices_labels_iob = {'O' : 0}
        self.indices_labels_iob_reverse = {0 : 'O'}
        
        for label_original in self.labels_original:
            label_b = label_original + '-B'
            label_i = label_original + '-I'
            
            for label in [label_b,label_i]:
                if label not in self.indices_labels_iob:
                    new_index = len(self.indices_labels_iob)
                    self.indices_labels_iob[label] = new_index
                    self.indices_labels_iob_reverse[new_index] = label


        
    def initialize_graph(self):
        self.session = tf.Session()
        K.set_session(self.session)
        self.session.run(tf.global_variables_initializer())
    
    
    
    def prepare(self):
        self.import_word_vectors()
        self.import_abstracts()
        self.extract_labels(self.abstracts_train)
        self.create_graph()
        self.create_feed_dicts()
        self.initialize_graph()
        
        
        
    def evaluate_taggings(self, taggings_gt, taggings_predicted):
        stats_exact = {}
        stats_partial = {}
        
        # initialization
        for label in self.labels_original:
            stats_exact[label] = F1Statistics()
            stats_partial[label] = F1Statistics()
            
        # evaluation
        for tagging_gt,tagging_predicted in zip(taggings_gt,taggings_predicted):
            for label in self.labels_original:
                # exact match
                set_gt = tagging_gt.to_set(label, True)
                set_predicted = tagging_predicted.to_set(label, True)
                
                stat = stats_exact[label]
                stat.num_occurences += len(set_gt)
                stat.true_positives += len(set_gt.intersection(set_predicted))
                stat.false_positives += len(set_predicted.difference(set_gt))
                
                # partial match
                set_gt = tagging_gt.to_set(label, False)
                set_predicted = tagging_predicted.to_set(label, False)
                
                stat = stats_partial[label]
                stat.num_occurences += len(set_gt)
                stat.true_positives += len(set_gt.intersection(set_predicted))
                stat.false_positives += len(set_predicted.difference(set_gt))
                
        return stats_partial, stats_exact
    
    
    
    def predict(self, feed_dict):
        [prediction] = self.session.run([self.taggings], feed_dict=feed_dict)
        return prediction
    
    
    
    def predict_annotated_abstract(self, abstract):
        annotated_abstract = AnnotatedAbstract()
        annotated_abstract.annotations = []
        
        sentences,feed_dicts = self.create_feed_dicts_from_abstracts([abstract])
        sentence_tokens = []
        sentence_numbers = []
        sentence_taggings = []
        
        for i,feed_dict in enumerate(feed_dicts):
            sentence_number = sentences[i].sentence_number
            tokens = abstract.get_sentence_tokens(sentence_number)
            sentence_tokens.append(tokens)
            sentence_numbers.append(sentence_number)
            
            tagging_encoded = self.predict(feed_dict)
            tagging_predicted = SentenceTagging()
            tagging_predicted.set_fom_encoding(tagging_encoded, self.labels_original, self.indices_labels_iob)
            sentence_taggings.append(tagging_predicted)
            
            
        new_annotations = convert_sentence_taggings_to_annotations(sentence_taggings,
                                                                   sentence_tokens,
                                                                   sentence_numbers)
            
        annotated_abstract.annotations = new_annotations
        annotated_abstract.assign_new_global_indices()
        
        return annotated_abstract
    
    
    def evaluate(self, feed_dicts):
        taggings_gt = []
        taggings_predicted = []
        
        for i,feed_dict in enumerate(feed_dicts):
            sentence = self.sentences_test[i]
            taggings_gt.append(sentence.sentence_tagging)
            
            [prediction] = self.session.run([self.taggings], feed_dict=feed_dict)
            tagging_predicted = SentenceTagging()
            tagging_predicted.set_fom_encoding(prediction, self.labels_original, self.indices_labels_iob)
            taggings_predicted.append(tagging_predicted)
            
        return self.evaluate_taggings(taggings_gt, taggings_predicted)
    
    
    
    def train_epoch(self):
        for feed_dict in self.feed_dicts_train:
            self.session.run([self.optimizer], feed_dict=feed_dict)
                
        
        
    def test(self):
        abstract_gt = self.abstracts_test[4]
        abstract_predicted = copy.deepcopy(abstract_gt)
        abstract_predicted.annotated_abstract = self.predict_annotated_abstract(abstract_gt)
        print_abstract_sentence(abstract_gt, 3)
        print('-----')
        print_abstract_sentence(abstract_predicted, 3)
        
        sys.exit()
        
        for abstract in self.abstracts_train:
            sentences.extend(self.extract_sentences_from_abtract(abstract))
        
        #sentences[5].print_out(self.indices_labels_iob, self.indices_labels_iob_reverse)
        #print('----------------')
        #print_abstract_sentence(self.abstracts_train[0], sentences[5].sentence_number)
        
        taggings_gt = []
        taggings_predicted = []
        
        for sentence in sentences:
            tagging = sentence.sentence_tagging
            encoded = tagging.encode(len(sentence.tokens), self.indices_labels_iob)
            tagging = SentenceTagging()
            tagging.set_fom_encoding(encoded, self.labels_original, self.indices_labels_iob)
            
            taggings_gt.append(sentence.sentence_tagging)
            taggings_predicted.append(tagging)
            
        stats_partial,stats_exact = self.evaluate_taggings(taggings_gt, taggings_predicted)
        print_stats(stats_exact)
        
        
        
    def train_validation(self, epochs):
        best_f1_validation = 0
        test_stats_collection = None
        worse_count = 0
        
        for epoch in range(epochs):
            print('epoch: ' + str(epoch+1))
            self.train_epoch()
            stats_partial,stats_exact = tagging_baseline.evaluate(tagging_baseline.feed_dicts_validation)
            f1 = compute_micro_stats(stats_exact).f1()
            
            if f1 > best_f1_validation:
                best_f1_validation = f1
                test_stats_partial, test_stats_exact = tagging_baseline.evaluate(tagging_baseline.feed_dicts_test)
                worse_count = 0
                
                # predict annotated abstracts ################################
                for abstract in tagging_baseline.abstracts_train:
                    abstract.annotated_abstract_predicted = tagging_baseline.predict_annotated_abstract(abstract)
                for abstract in tagging_baseline.abstracts_validation:
                    abstract.annotated_abstract_predicted = tagging_baseline.predict_annotated_abstract(abstract)
                for abstract in tagging_baseline.abstracts_test:
                    abstract.annotated_abstract_predicted = tagging_baseline.predict_annotated_abstract(abstract)
            else:
                worse_count += 1
                
        return test_stats_partial, test_stats_exact

    

tagging_baseline = TaggingBasline()
tagging_baseline.prepare()
tagging_baseline.print_label_frequency(tagging_baseline.abstracts_test)
sys.exit()

stats_partial,stats_exact = tagging_baseline.train_validation(30)
print_stats(stats_exact)


'''
f = open('diabetes_slot_filling_pipeline.dump', 'wb')
triple = (tagging_baseline.abstracts_train, tagging_baseline.abstracts_validation, tagging_baseline.abstracts_test)
pickle.dump(triple, f)
f.close()
'''


sorted_class_names = sorted(stats_partial.keys())
string = ''

for class_name in sorted_class_names:
    precision_partial = stats_partial[class_name].precision()
    recall_partial = stats_partial[class_name].recall()
    f1_partial = stats_partial[class_name].f1()
    
    precision_exact = stats_exact[class_name].precision()
    recall_exact = stats_exact[class_name].recall()
    f1_exact = stats_exact[class_name].f1()
    
    string += class_name + ' & '
    
    # partial match data
    string += format(precision_partial, '0.2f') + ' & '
    string += format(recall_partial, '0.2f') + ' & '
    string += format(f1_partial, '0.2f') + ' & '
    
    # exact match
    string += format(precision_exact, '0.2f') + ' & '
    string += format(recall_exact, '0.2f') + ' & '
    string += format(f1_exact, '0.2f')
    
    string += ' \\\\ \n'
    
micro_stats_partial = compute_micro_stats(stats_partial)
micro_stats_exact = compute_micro_stats(stats_exact)

string += '\\midrule \\midrule \n'
string += '& '
string += format(micro_stats_partial.precision(), '0.2f') + ' & '
string += format(micro_stats_partial.recall(), '0.2f') + ' & '
string += format(micro_stats_partial.f1(), '0.2f') + ' & '

string += format(micro_stats_exact.precision(), '0.2f') + ' & '
string += format(micro_stats_exact.recall(), '0.2f') + ' & '
string += format(micro_stats_exact.f1(), '0.2f')

string += ' \\\\ \n'
    
    
print(string)
sys.exit()


# save stats
f = open('stat_collections/' + SETTING_DISEASE + '_tagging.dump', 'wb')
pickle.dump((stats_partial,stats_exact),f)
f.close()


        