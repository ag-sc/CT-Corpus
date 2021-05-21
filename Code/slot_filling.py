#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.layers import Embedding,Conv1D,Dense,Dropout
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.initializers import Constant
import numpy as np
import random
import itertools
import copy
import sys
import pickle
from ctro import *

PATIENCE = 1
MIN_SLOT_FREQ = 10
kernel_size_entities = 2
kernel_size_context = 3


SETTING_PIPELINE = False
SETTING_WORD_VECTORS = 'WIKIPEDIA'
SETTING_DISEASE = sys.argv[1]
SETTING_CONTEXT = sys.argv[2] # COMPLETE, KEY, NONE

if SETTING_DISEASE != 'gl' and SETTING_DISEASE != 'dm2':
    print('Invalid disease parameter!')
    sys.exit()
if SETTING_CONTEXT not in ['complete', 'key', 'none', 'heuristic']:
    print('Invalid context paramter!')
    sys.exit()
if sys.argv[3] == 'pipeline':
    SETTING_PIPELINE = True
    
    

# string constants
NUM_FILTERS_CONTEXT = 'num_filters_context'
NUM_FILTERS_ENTITIES = 'num_filters_entities'
HIDDEN_LAYER_DIM_ENTITIES = 'hidden_layer_dim_entities'


if SETTING_WORD_VECTORS == 'WIKIPEDIA':
    WORD_VECTORS_PATH = '/homes/cwitte/word_vectors/glove_wikipedia/glove.6B.100d.txt'
else:
    WORD_VECTORS_PATH = '/homes/cwitte/word_vectors/pmc_pubmed_wikipedia/word_vectors.txt'

if SETTING_DISEASE == 'gl':
    if SETTING_PIPELINE:
        DATA_DUMP_PATH = 'glaucoma_slot_filling_pipeline.dump'
    else:
        DATA_DUMP_PATH = 'glaucoma_slot_filling.dump'
elif SETTING_DISEASE == 'dm2':
    if SETTING_PIPELINE:
        DATA_DUMP_PATH = 'diabetes_slot_filling_pipeline.dump'
    else:
        DATA_DUMP_PATH = 'diabetes_slot_filling.dump'
    

# settings ###################################################################
SETTING_SLOT_FILLING_ONLY = 1
SETTING = SETTING_SLOT_FILLING_ONLY

class EntityConvLayer(tf.keras.layers.Layer):
    def __init__(self, num_filters_entities, kernel_size_entities, num_filters_context, kernel_size_context):
        super(EntityConvLayer, self).__init__()
        
        self.conv_layer_entities = Conv1D(filters=num_filters_entities, kernel_size=kernel_size_entities, padding='same')
        self.conv_layer_context = Conv1D(filters=num_filters_context, kernel_size=kernel_size_context, padding='same')
        
        
    def call(self, entities, left_contexts, right_contexts):
        # convolutions ####################################################################################
        entities_convolved = self.conv_layer_entities(entities)
        left_contexts_convolved = self.conv_layer_context(left_contexts)
        right_contexts_convolved = self.conv_layer_context(right_contexts)
        
        # exchange last two dimensions: (timesteps,filters) => (filters,timesteps) ###########################
        entities_convolved = tf.transpose(entities_convolved, perm=[0,2,1]) 
        left_contexts_convolved = tf.transpose(left_contexts_convolved, perm=[0,2,1])
        right_contexts_convolved = tf.transpose(right_contexts_convolved, perm=[0,2,1])
        
        # max pooling ########################################################################################
        entities_pooled = tf.reduce_max(entities_convolved, axis=-1)
        left_contexts_pooled = tf.reduce_max(left_contexts_convolved, axis=-1)
        right_contexts_pooled = tf.reduce_max(right_contexts_convolved, axis=-1)
    
        # create feature vectors for entities ###############################################################
        entity_features = tf.concat([left_contexts_pooled, entities_pooled, right_contexts_pooled], axis=-1)
        
        return entity_features




class SlotFilling:
    def __init__(self):
        self.train = None
        self.validation = None
        self.test = None
        self.used_slots = None
        
        self.session = None
        
        
    def import_abstracts(self):
        f = open(DATA_DUMP_PATH, 'rb')
        self.abstracts_train, self.abstracts_validation, self.abstracts_test = pickle.load(f, encoding='latin1')
        f.close()
        
        
        
    def estimate_used_slots(self):
        slot_counts = {slot_name:0 for slot_name in ontology.used_slots}
        self.used_slots = set()
        
        for abstract in self.abstracts_train:
            group_collection = abstract.group_collection
            
            for group_name in ontology.used_group_names:
                for i in range(len(group_collection.groups[group_name])):
                    group = group_collection.groups[group_name][i]
                    
                    for slot_name in group.slots:
                        slot_counts[slot_name] += 1
                        
        for slot_name in slot_counts:
            if slot_counts[slot_name] >= MIN_SLOT_FREQ:
                self.used_slots.add(slot_name)
        
        
        
    # slot name -> reference slot name
    def estimate_reference_slots(self, abstracts):
        stats = {}
        
        for group_name in ontology.group_names:
            stats[group_name] = {}
            
            for slot_name_outer in ontology.group_slots[group_name]:
                stats[group_name][slot_name_outer] = {}
                
                for slot_name_inner in ontology.group_slots[group_name]:
                    stats[group_name][slot_name_outer][slot_name_inner] = 0
    
        
        # create co-occurence statistics
        for abstract in abstracts:
            group_collection = abstract.group_collection
                
            for group_name in group_collection.groups:
                for group in group_collection.groups[group_name]:
                    for slot_name_outer in group.slots:
                        for slot_name_inner in group.slots:
                            if slot_name_outer == slot_name_inner:
                                continue
                        
                            stats[group_name][slot_name_inner][slot_name_outer] += 1
                            stats[group_name][slot_name_outer][slot_name_inner] += 1
                    
        
        # estimate reference slots
        reference_slots = {}
        
        for group_name in ontology.group_names:
            for query_slot_name in ontology.group_slots[group_name]:
                if query_slot_name not in self.used_slots:
                    continue
                
                reference_slot_name = None
                max_count = 0
                
                for slot_name in ontology.group_slots[group_name]:
                    if slot_name == query_slot_name:
                        continue
                    if slot_name not in self.used_slots:
                        continue
                    
                    count = stats[group_name][query_slot_name][slot_name]
                    if count > max_count:
                        reference_slot_name = slot_name
                        max_count = count
                
                if reference_slot_name is not None:
                    reference_slots[query_slot_name] = reference_slot_name
                
        self.reference_slots = reference_slots
        
        
        
    def get_reference_global_annotation_indices(self, query_slot_name, group):
        if query_slot_name not in self.used_slots:
            return []
        if query_slot_name not in self.reference_slots:
            return []

        reference_slot_name = self.reference_slots[query_slot_name]
        reference_indices = []
        
        if reference_slot_name not in group.slots:
            return []
        
        for slot_value in group.slots[reference_slot_name]:
            annotation = slot_value.annotation
            
            if annotation is not None:
                reference_indices.append(annotation.global_index)
                
        return reference_indices
        
        
        
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
    
    
    def create_graph(self):
        # hyperparameters #########################################################
        num_filters_context = 600
        num_filters_entities = 100
        hidden_layer_dim_entities = 50
        hidden_layer_dim_reference_entities = 50
        hidden_layer_dim_value_entities = 50
        hidden_layer_dim_slot_candidates = 50
    
        # placeholders ############################################################
        # entity strings
        self.placeholder_entities_gt = tf.placeholder(shape=(None,None), dtype=tf.float32)
        self.placeholder_left_contexts_gt = tf.placeholder(shape=(None,None), dtype=tf.float32)
        self.placeholder_right_contexts_gt = tf.placeholder(shape=(None,None), dtype=tf.float32)
        
        self.placeholder_entities_predicted = tf.placeholder(shape=(None,None), dtype=tf.float32)
        self.placeholder_left_contexts_predicted = tf.placeholder(shape=(None,None), dtype=tf.float32)
        self.placeholder_right_contexts_predicted = tf.placeholder(shape=(None,None), dtype=tf.float32)
        
        # entity labels
        self.placeholder_entity_labels = tf.placeholder(shape=(None,None), dtype=tf.float32)
        
        # slot labels and indices for each group
        self.placeholders_slot_multi_labels = {}
        self.placeholders_reference_indices = {}
        self.placeholders_active_slots_masks_gt = {}
        self.placeholders_active_slots_masks_predicted = {}
        self.placeholders_group_masks = {}
        
        # vars ####################################################################
        self.var_context = tf.Variable(tf.zeros((1,hidden_layer_dim_value_entities)), dtype=tf.float32)
        
        
        for group_name in ontology.used_group_names:
            self.placeholders_slot_multi_labels[group_name] = tf.placeholder(shape=(None,None,None), dtype=tf.float32)
            self.placeholders_active_slots_masks_gt[group_name] = tf.placeholder(shape=(None,None,None), dtype=tf.float32)
            self.placeholders_active_slots_masks_predicted[group_name] = tf.placeholder(shape=(None,None,None), dtype=tf.float32)
            self.placeholders_group_masks[group_name] = tf.placeholder(shape=(None), dtype=tf.float32)
            self.placeholders_reference_indices[group_name] = {}
            
            for slot_name in ontology.group_slots[group_name]:
                if slot_name not in self.used_slots:
                    continue
                
                self.placeholders_reference_indices[group_name][slot_name] = tf.placeholder(shape=(None,None), dtype=tf.float32)
        
        # learning rate/phase
        self.placeholder_learning_rate = tf.placeholder(shape=(), dtype=tf.float32)
        self.placeholder_learning_phase = tf.placeholder(shape=(), dtype=tf.float32)
        
        # number of group instances ##########################################################################
        num_group_instances = {}
        for group_name in ontology.used_group_names:
            num_group_instances[group_name] = tf.shape(self.placeholders_slot_multi_labels[group_name])[0]
            
        if SETTING_PIPELINE:
            num_entities_predicted = tf.shape(self.placeholder_entities_predicted)[0]
            num_entities_gt = tf.shape(self.placeholder_entities_gt)[0]
        else:
            num_entities_predicted = tf.shape(self.placeholder_entities_gt)[0]
            num_entities_gt = tf.shape(self.placeholder_entities_gt)[0]
            
        # layers ########################################################################################
        # embedding layer
        embedding_layer = Embedding(self.vocab_size+3, self.embedding_dim, trainable=False)
        embedding_layer.build(input_shape=(1,)) # the input_shape here has no effect in the build function
        embedding_layer.set_weights([self.embedding_matrix])
        
        # convolution layers
        conv_layer = EntityConvLayer(num_filters_entities, kernel_size_entities, num_filters_context, kernel_size_context)
        
        # hidden layers
        hidden_layer_entities = Dense(hidden_layer_dim_entities, activation='tanh')
        hidden_layer_value_entities = Dense(hidden_layer_dim_value_entities, activation='tanh')
        hidden_layer_reference_entities = Dense(hidden_layer_dim_reference_entities, activation='tanh')
        hidden_layer_slot_candidates = Dense(hidden_layer_dim_slot_candidates, activation='tanh')

        # final linear layers
        logits_layer_entities = Dense(len(ontology.used_most_general_classes_indices))
        
        logits_layers_slot_candidates = {}
        for group_name in ontology.used_group_names:
            logits_layers_slot_candidates[group_name] = {}
            
            for slot_name in ontology.slot_indices[group_name]:
                logits_layers_slot_candidates[group_name][slot_name] = Dense(1)
        
        dropout = Dropout(0.5)
        
        # apply embedding layer ##################################################################
        entities_gt = embedding_layer(self.placeholder_entities_gt)
        left_contexts_gt = embedding_layer(self.placeholder_left_contexts_gt)
        right_contexts_gt = embedding_layer(self.placeholder_right_contexts_gt)
        
        entities_predicted = embedding_layer(self.placeholder_entities_predicted)
        left_contexts_predicted = embedding_layer(self.placeholder_left_contexts_predicted)
        right_contexts_predicted = embedding_layer(self.placeholder_right_contexts_predicted)
    
        # create feature vectors for entities ###############################################################
        entity_features_gt = conv_layer(entities_gt, left_contexts_gt, right_contexts_gt)
        entity_features_predicted = conv_layer(entities_predicted, left_contexts_predicted, right_contexts_predicted)
        
        # entity representation
        value_entities_hidden_representation_gt = hidden_layer_value_entities(entity_features_gt)
        value_entities_hidden_representation_predicted = hidden_layer_value_entities(entity_features_predicted)
        
        # add dummy vector to reference entities for groups which
        # have only one slot set, i.e. no context
        reference_entities_hidden_representation = tf.concat([value_entities_hidden_representation_gt, self.var_context], axis=0)
        
        # compute representation of context #################################################################
        # retrieve context entities for each slot
        slot_contexts = {}
        for group_name in ontology.used_group_names:
            slot_contexts[group_name] = {}
            
            for slot_name in ontology.slot_indices[group_name]:
                if slot_name not in self.used_slots:
                    continue
                if SETTING_CONTEXT == 'complete':
                    mask = tf.one_hot(ontology.slot_indices[group_name][slot_name], depth=len(ontology.slot_indices[group_name]))
                    mask = tf.expand_dims(mask, axis=0)
                    mask = tf.expand_dims(mask, axis=0)
                    
                    # invert mask
                    mask = tf.greater(mask, 0)
                    mask = tf.math.logical_not(mask)
                    mask = tf.cast(mask, tf.float32)
                    
                    mask = tf.tile(mask, [num_group_instances[group_name], num_entities_gt, 1])
                    mask = tf.multiply(mask, self.placeholders_slot_multi_labels[group_name])
                    mask = tf.reduce_sum(mask, axis=-1)
                    mask = tf.greater(mask, 0)
                elif SETTING_CONTEXT == 'key':
                    mask = self.placeholders_reference_indices[group_name][slot_name]
                    mask = tf.greater(mask, 0)
                
                if SETTING_CONTEXT != 'none':
                    trues = tf.constant([[True]])
                    trues = tf.tile(trues, [num_group_instances[group_name], 1])
                    mask = tf.concat([mask,trues], axis=-1)
                    
                    context_entities = tf.expand_dims(reference_entities_hidden_representation, axis=0)
                    context_entities = tf.tile(context_entities, [num_group_instances[group_name], 1, 1])
                    
                    context = tf.ragged.boolean_mask(context_entities, mask)
                    
                    # aggregation
                    context = tf.reduce_mean(context, axis=-2)
                    
                    # save
                    slot_contexts[group_name][slot_name] = context
        
        # compute hidden representation ######################################################################  
        # slot fillers
        slot_candidates_hidden_representations = {}
        '''
        for group_name in ontology.used_group_names:
            slot_candidates = tf.expand_dims(entity_features, axis=0)
            slot_candidates = tf.tile(slot_candidates, [num_group_instances[group_name], 1, 1])
            slot_candidates_hidden_representations[group_name] = hidden_layer_slot_candidates(slot_candidates)
        '''
        for group_name in ontology.used_group_names:
            slot_candidates_hidden_representations[group_name] = {}
            
            if SETTING_PIPELINE:
                value_entities = tf.expand_dims(value_entities_hidden_representation_predicted, axis=0)
            else:
                value_entities = tf.expand_dims(value_entities_hidden_representation_gt, axis=0)
                
            value_entities = tf.tile(value_entities, [num_group_instances[group_name], 1, 1])
            
            for slot_name in ontology.slot_indices[group_name]:
                if SETTING_CONTEXT != 'none':
                    if slot_name not in self.used_slots:
                        context = tf.zeros((num_group_instances[group_name], hidden_layer_dim_value_entities))
                    else:
                        context = slot_contexts[group_name][slot_name]
                        
                    context = tf.expand_dims(context, axis=1)
                    context = tf.tile(context, [1, num_entities_predicted, 1])
                
                    representation = tf.concat([value_entities, context], axis=-1)
                    representation = tf.reshape(representation, (num_group_instances[group_name], num_entities_predicted, 2*hidden_layer_dim_value_entities))
                else:
                    representation = value_entities
                    
                representation = hidden_layer_slot_candidates(representation)
                self.debug = representation
                slot_candidates_hidden_representations[group_name][slot_name] = representation
        
        
        # apply droput layer ################################################################################
        # entities
        #entity_hidden_representations = dropout(entity_hidden_representations)
        
        # slot candidates
        #for group_name in ontology.used_group_names:
            #slot_candidates_hidden_representations[group_name] = dropout(slot_candidates_hidden_representations[group_name])
        
        
        # compute logits #######################################################################################
        # entities
        logits_entities = logits_layer_entities(value_entities_hidden_representation_gt)
        self.logits_entities = logits_entities
        
        # slot candidates
        logits_slot_candidates = {}
        for group_name in ontology.used_group_names:
            logits_slot_candidates[group_name] = {}
            
            for slot_name in ontology.slot_indices[group_name]:
                logits_slot_candidates[group_name][slot_name] = logits_layers_slot_candidates[group_name][slot_name](slot_candidates_hidden_representations[group_name][slot_name])
            
        # stack logits
        for group_name in ontology.used_group_names:
            logits_list = []
            
            for i in range(len(ontology.slot_indices[group_name])):
                slot_name = ontology.slot_indices_reverse[group_name][i]
                logits_list.append(logits_slot_candidates[group_name][slot_name])
                
            logits_slot_candidates[group_name] = tf.concat(logits_list,  axis=-1)
            
            
        # predict slot fillers #######################################################################
        prediction_slot_fillers = {}
        for group_name in ontology.used_group_names:
            prediction = tf.nn.sigmoid(logits_slot_candidates[group_name])
            prediction = tf.math.greater(prediction, 0.5)
            prediction = tf.cast(prediction, tf.float32)
            prediction_slot_fillers[group_name] = prediction
            
            # masking
            if SETTING == SETTING_SLOT_FILLING_ONLY:
                prediction_slot_fillers[group_name] = tf.multiply(prediction_slot_fillers[group_name], self.placeholders_active_slots_masks_predicted[group_name])
            
        self.prediction_slot_fillers = prediction_slot_fillers
        
            
        # loss ###############################################################################################
        # entities
        crossent_entities = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.placeholder_entity_labels, logits=logits_entities)
        
        # slot candidates
        crossent_slot_candidates = tf.constant(0.0)
        for group_name in ontology.used_group_names:
            loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_slot_candidates[group_name], labels=self.placeholders_slot_multi_labels[group_name])
                
            # masking
            if SETTING == SETTING_SLOT_FILLING_ONLY:
                loss = tf.multiply(loss, self.placeholders_active_slots_masks_predicted[group_name])
            
            loss = tf.reduce_sum(loss, axis=-1)
            loss = tf.reduce_sum(loss, axis=-1)
            
            # mask dummy groups
            loss = tf.multiply(loss, self.placeholders_group_masks[group_name])
            
            loss = tf.reduce_sum(loss)
            crossent_slot_candidates += loss
            
        self.loss_slot_candidates = crossent_slot_candidates
        
        # optimizer ##########################################################################################
        # entities
        self.optimizer_entities = tf.train.GradientDescentOptimizer(self.placeholder_learning_rate).minimize(crossent_entities )
        
        # slot candidates
        self.optimizer_slot_candidates = tf.train.GradientDescentOptimizer(self.placeholder_learning_rate).minimize(crossent_slot_candidates )
    
    
    
    def encode_annotations(self, annotations):
        entities = []
        left_contexts = []
        right_contexts = []
        
        for annotation in annotations:
            entity = [self.encode_token(token) for token in annotation.tokens]
            left_context = [self.encode_token(token) for token in annotation.left_context]
            right_context = [self.encode_token(token) for token in annotation.right_context]
            
            if len(entity) == 0:
                entity.append(self.encode_token('None'))
            if len(left_context) == 0:
                left_context.append(self.encode_token('None'))
            if len(right_context) == 0:
                right_context.append(self.encode_token('None'))
            
            entities.append(entity)
            left_contexts.append(left_context)
            right_contexts.append(right_context)
            
        # zero padding
        max_len_entities = max([len(entity) for entity in entities])
        max_len_left_contexts = max([len(left_context) for left_context in left_contexts])
        max_len_right_contexts = max([len(right_context) for right_context in right_contexts])
        
        entities = [entity + [0]*(max_len_entities - len(entity)) for entity in entities]
        left_contexts = [left_context + [0]*(max_len_left_contexts - len(left_context)) for left_context in left_contexts]
        right_contexts = [right_context + [0]*(max_len_right_contexts - len(right_context)) for right_context in right_contexts]
        
        return left_contexts, entities, right_contexts
    
        
        
    def create_feed_dict_abstract(self, abstract, bool_training_abstract):
        annotations = abstract.annotated_abstract.annotations
        feed_dict = {}
        
        # encoded strings
        left_contexts_gt, entities_gt, right_contexts_gt = self.encode_annotations(annotations)
        
        if abstract.annotated_abstract_predicted != None:
            annotations_predicted = abstract.annotated_abstract_predicted.annotations
            left_contexts_predicted, entities_predicted, right_contexts_predicted = self.encode_annotations(annotations_predicted)
        else:
            left_contexts_predicted = [[0]]
            entities_predicted = [[0]]
            right_contexts_predicted = [[0]]
        
        # encode labels ##############################################################################
        # entity labels
        entity_labels = []
        for annotation in annotations:
            index = ontology.used_most_general_classes_indices[annotation.most_general_superclass]
            vec = np.zeros((len(ontology.used_most_general_classes_indices)))
            vec[index] = 1
            entity_labels.append(vec)
            
        # slot labels
        encoded_groups = abstract.group_collection.encode(abstract.annotated_abstract)
        for group_name in ontology.used_group_names:
            num_groups = len(abstract.group_collection.groups[group_name])
            
            if num_groups == 0:
                data = np.zeros((1, len(annotations), len(ontology.slot_indices[group_name])))
                feed_dict[self.placeholders_slot_multi_labels[group_name]] = data
            else:
                feed_dict[self.placeholders_slot_multi_labels[group_name]] = encoded_groups[group_name]
                
                
        # reference indices #################################################################################
        for group_name in ontology.used_group_names:
            num_groups = len(abstract.group_collection.groups[group_name])
            
            for slot_name in ontology.group_slots[group_name]:
                if slot_name not in self.used_slots:
                    continue
                
                reference_indices_matrix = np.zeros( (num_groups, len(annotations)) )
                
                for i in range(num_groups):
                    group = abstract.group_collection.groups[group_name][i]
                    reference_indices = self.get_reference_global_annotation_indices(slot_name, group)
                    
                    for reference_index in reference_indices:
                        reference_indices_matrix[i,reference_index] = 1
                
                if num_groups == 0: # no group instance -> dummy data
                    reference_indices_matrix = np.zeros( (1, len(annotations)) )
                    
                feed_dict[self.placeholders_reference_indices[group_name][slot_name]] = reference_indices_matrix
            
            
        # masks #########################################################################################
        # group masks
        for group_name in ontology.used_group_names:
            num_groups = len(abstract.group_collection.groups[group_name])
            
            if num_groups == 0:
                vec = [0]
            else:
                vec = [1] * num_groups
                
            feed_dict[self.placeholders_group_masks[group_name]] = vec
            
        # active slots masks
        if abstract.annotated_abstract_predicted != None:
            annotations_predicted = abstract.annotated_abstract_predicted.annotations
        else:
            annotations_predicted = annotations
            
        for group_name in ontology.used_group_names:
            num_groups = len(abstract.group_collection.groups[group_name])
            num_slots = len(ontology.slot_indices[group_name])
            
            if num_groups == 0: # dummy data
                masks_gt = np.zeros((1, len(annotations), num_slots))
                masks_predicted = np.zeros((1, len(annotations_predicted), num_slots))
            else:
                masks_gt = []
                masks_predicted = []
                
                for i in range(num_groups):
                    group_masks_gt = []
                    group_masks_predicted = []
                    
                    for annotation in annotations:
                        class_name = annotation.most_general_superclass
                        group_masks_gt.append(ontology.active_slot_masks[group_name][class_name])
                        
                    for annotation_predicted in annotations_predicted:
                        class_name = annotation_predicted.most_general_superclass
                        group_masks_predicted.append(ontology.active_slot_masks[group_name][class_name])
                        
                    masks_gt.append(group_masks_gt)
                    masks_predicted.append(group_masks_predicted)
                    
            feed_dict[self.placeholders_active_slots_masks_gt[group_name]] = masks_gt
            if bool_training_abstract:
                feed_dict[self.placeholders_active_slots_masks_predicted[group_name]] = masks_gt
            else:
                feed_dict[self.placeholders_active_slots_masks_predicted[group_name]] = masks_predicted
        
        # create feed dict ###########################################
        # encoded strings
        feed_dict[self.placeholder_entities_gt] = entities_gt
        feed_dict[self.placeholder_left_contexts_gt] = left_contexts_gt
        feed_dict[self.placeholder_right_contexts_gt] = right_contexts_gt
        
        if bool_training_abstract:
            feed_dict[self.placeholder_entities_predicted] = entities_gt
            feed_dict[self.placeholder_left_contexts_predicted] = left_contexts_gt
            feed_dict[self.placeholder_right_contexts_predicted] = right_contexts_gt
        else:
            feed_dict[self.placeholder_entities_predicted] = entities_predicted
            feed_dict[self.placeholder_left_contexts_predicted] = left_contexts_predicted
            feed_dict[self.placeholder_right_contexts_predicted] = right_contexts_predicted
        
        # labels
        feed_dict[self.placeholder_entity_labels] = entity_labels
        
        # default learning rate/phase
        feed_dict[K.learning_phase()] = 1
        feed_dict[self.placeholder_learning_phase] = 1
        feed_dict[self.placeholder_learning_rate] = 0.01
        
        return feed_dict
    
    
    def create_feed_dicts_all_abstracts(self):
        # create feed dicts
        self.train = []
        self.validation = []
        self.test = []
        
        for abstract in self.abstracts_train:
            feed_dict = self.create_feed_dict_abstract(abstract, True)
            self.train.append((abstract, feed_dict))
            
        for abstract in self.abstracts_validation:
            feed_dict = self.create_feed_dict_abstract(abstract, False)
            self.validation.append((abstract, feed_dict))
            
        for abstract in self.abstracts_test:
            feed_dict = self.create_feed_dict_abstract(abstract, False)
            self.test.append((abstract, feed_dict))
            
            
        
    def initialize_graph(self):
        self.session = tf.Session()
        K.set_session(self.session)
        self.session.run(tf.global_variables_initializer())
        
        
    def print_reference_slots(self):
        global DATA_DUMP_PATH
        DATA_DUMP_PATH = 'glaucoma_slot_filling.dump'
        self.import_abstracts()
        self.estimate_used_slots()
        self.estimate_reference_slots(self.abstracts_train)
        ref_slots_glaucoma = self.reference_slots
        
        DATA_DUMP_PATH = 'diabetes_slot_filling.dump'
        self.import_abstracts()
        self.estimate_used_slots()
        self.estimate_reference_slots(self.abstracts_train)
        ref_slots_diabetes = self.reference_slots
        
        query_slots_glaucoma = set(ref_slots_glaucoma.keys())
        query_slots_diabetes = set(ref_slots_diabetes.keys())
        query_slots = query_slots_glaucoma.union(query_slots_diabetes)
        
        string = ''
        for query_slot in sorted(query_slots):
            string += query_slot + ' & '
            
            if query_slot in ref_slots_glaucoma:
                string += ref_slots_glaucoma[query_slot] + ' & '
            else:
                string += '- & '
                
            if query_slot in ref_slots_diabetes:
                string += ref_slots_diabetes[query_slot]
            else:
                string += '- '
                
            string += ' \\\\\n'
        
        print(string)
    
    
    def prepare(self):
        self.import_word_vectors()
        self.import_abstracts()
        self.estimate_used_slots()
        self.estimate_reference_slots(self.abstracts_train)
        
        self.create_graph()
        self.create_feed_dicts_all_abstracts()
        self.initialize_graph()
    
    
    def train_abstract(self, feed_dict_abstract, optimizer_node, learning_rate):
        feed_dict_abstract[K.learning_phase()] = 1
        feed_dict_abstract[self.placeholder_learning_phase] = 1
        feed_dict_abstract[self.placeholder_learning_rate] = learning_rate
        
        self.session.run([optimizer_node], feed_dict=feed_dict_abstract)
        
        
    def train_epoch(self, optimizer_node, learning_rate):
        for abstract,feed_dict in self.train:
            self.train_abstract(feed_dict, optimizer_node, learning_rate)
            
            
    def evalute_entities(self):
        stats_dict = {}
        for class_name in ontology.used_most_general_classes_indices:
            stats_dict[class_name] = F1Statistics()
            
        # evalutae
        for abstract,feed_dict in self.test:
            feed_dict[K.learning_phase()] = 0
            feed_dict[self.placeholder_learning_phase] = 0
        
            [logits_abstract] = self.session.run([self.logits_entities], feed_dict=feed_dict)
            for i,_ in enumerate(logits_abstract):
                # prediction
                argmax_predicted = np.argmax(logits_abstract[i])
                predicted_class_name = ontology.used_most_general_classes_indices_reverse[argmax_predicted]
                
                # ground truth
                argmax_gt = np.argmax(feed_dict[self.placeholder_entity_labels][i])
                gt_class_name = ontology.used_most_general_classes_indices_reverse[argmax_gt]
                stats_dict[gt_class_name].num_occurences += 1
                
                if predicted_class_name == gt_class_name:
                    stats_dict[predicted_class_name].true_positives += 1
                else:
                    stats_dict[predicted_class_name].false_positives += 1
                    
        return stats_dict
    
    
    
    def predict_group_collection(self, feed_dict):
        feed_dict[K.learning_phase()] = 0
        feed_dict[self.placeholder_learning_phase] = 0
        
        query = []
        for group_name in ontology.used_group_names:
            query.append(self.prediction_slot_fillers[group_name])
            
        prediction_list = self.session.run(query, feed_dict=feed_dict)
        
        prediction_dict = {}
        for i,group_name in enumerate(ontology.used_group_names):
            prediction_dict[group_name] = prediction_list[i].tolist()
            
        return prediction_dict
            
    
    
    def evaluate_slot_filling(self, bool_validation=False):
        stats_collection = F1StatisticsCollection()
        
        if bool_validation:
            data = self.validation
        else:
            data = self.test
            
        for abstract,feed_dict in data:
            prediction = self.predict_group_collection(feed_dict)
            group_collection_predicted = GroupCollection()
            if not SETTING_PIPELINE:
                group_collection_predicted.set_from_encoded_groups(prediction, abstract.annotated_abstract)
            else:
                group_collection_predicted.set_from_encoded_groups(prediction, abstract.annotated_abstract_predicted)
            
            # update statistics
            for group_name in ontology.used_group_names:
                for i in range(len(abstract.group_collection.groups[group_name])):
                    group_gt = abstract.group_collection.groups[group_name][i]
                    group_prediction = group_collection_predicted.groups[group_name][i]
                    stats_collection.update(group_gt, group_prediction, self.used_slots)
                    
        return stats_collection
    
    
    def get_loss_slot_candidates(self):
        loss_sum = 0.0
        
        for abstract,feed_dict in self.train:
            [loss] = self.session.run([self.loss_slot_candidates], feed_dict=feed_dict)
            loss_sum += loss
            
        return loss_sum
    
    
    def get_debug_info(self):
        abstract,feed_dict = self.train[0]
        [debug] = self.session.run([self.debug], feed_dict=feed_dict)
        return debug
    
    
    def train_validation(self, initial_learning_rate, patience, max_epochs=None):
        best_f1_validation = 0
        test_stats_collection = None
        worse_count = 0
        epoch = 0
        
        while True:
            self.train_epoch(slot_filling.optimizer_slot_candidates, initial_learning_rate)
            f1 = self.evaluate_slot_filling(True).get_micro_stats().f1()
            
            if f1 > best_f1_validation:
                best_f1_validation = f1
                test_stats_collection = self.evaluate_slot_filling()
                worse_count = 0
                
                # print stats
                micro_stats = test_stats_collection.get_micro_stats()
                for group_name in ontology.used_group_names:
                    group_micro_stats = test_stats_collection.get_micro_stats_group(group_name)
                    
                    print(group_name + '(precision, recall, f1):')
                    print(str(group_micro_stats.precision()) + ' ' + str(group_micro_stats.recall()) + ' ' + str(group_micro_stats.f1()))
                    print(' ')
                
                print('micro f1: ' + str(micro_stats.f1()))
                test_stats_collection.print_common_error_statistics()
                print('-----------------------------')
            else:
                worse_count += 1
                
            if worse_count >= patience:
                return test_stats_collection
            
            epoch += 1
            print('epoch: ' + str(epoch))
            if max_epochs != None and epoch >= max_epochs:
                return test_stats_collection
            
            
    
    def get_closest_annotation(self, all_annotations, reference_doc_char_onset, query_slot_name, b_pipeline_setting):
        annotations = []
        query_slot_type = ontology.slot_type[query_slot_name]
        
        for annotation in all_annotations:
            if b_pipeline_setting:
                if ontology.most_general_superclass[query_slot_type] == annotation.most_general_superclass:
                    annotations.append(annotation)
            else:
                if annotation.label in ontology.all_subclasses[query_slot_type]:
                    annotations.append(annotation)
        
        annotations_distance = [(annotation, abs(annotation.doc_char_onset-reference_doc_char_onset)) for annotation in annotations if annotation.doc_char_onset != reference_doc_char_onset]
        if len(annotations_distance) == 0:
            return None
        
        return sorted(annotations_distance, key=lambda x: x[1])[0][0]
        
    
    
    def heuristic(self):
        stats_collection = F1StatisticsCollection()

        for abstract,_ in self.test:
            #print('abstract ' + abstract.abstract_id + ' --------------------------')
            annotated_abstract = abstract.annotated_abstract
            group_collection_gt = abstract.group_collection
            group_collection_predicted = group_collection_gt.create_structure_copy()
            
            for group_name in group_collection_gt.groups:
                for i in range(len(group_collection_gt.groups[group_name])):
                    group_gt = group_collection_gt.groups[group_name][i]
                    group_predicted = group_collection_predicted.groups[group_name][i]
                    
                    for query_slot_name in group_gt.slots:
                        if query_slot_name not in self.reference_slots:
                            continue
                        
                        reference_slot_name = self.reference_slots[query_slot_name]
                        
                        if reference_slot_name not in self.used_slots:
                            continue
                        if reference_slot_name not in group_gt.slots:
                            continue
                        
                        reference_slot_values = group_gt.slots[reference_slot_name]
                        reference_annotations = [slot_value.annotation for slot_value in group_gt.slots[reference_slot_name]]
                        if len(reference_annotations) == 0:
                            continue
                        
                        reference_annotation = random.choice(reference_annotations)
                        reference_doc_char_onset = reference_annotation.doc_char_onset
                        all_annotations = annotated_abstract.annotations
                        
                        closest_annotation = self.get_closest_annotation(all_annotations,
                                                                         reference_doc_char_onset,
                                                                         query_slot_name,
                                                                         SETTING_PIPELINE)
                        
                        if closest_annotation is None:
                            continue

                        slot_value = SlotValue()
                        slot_value.string = ' '.join(closest_annotation.tokens)
                        slot_value.annotation = closest_annotation
                        group_predicted.add_value(query_slot_name, slot_value)
                    
                    stats_collection.update(group_gt, group_predicted)
                    
                    '''
                    for slot_name in slot_list:
                        if slot_name not in group_gt.slots:
                            continue
                        
                        print('slot type: ' + slot_name)
                        print('predicted: ' + set_to_string(group_predicted.slot_values_as_set(slot_name)))
                        print('ground truth: ' + set_to_string(group_gt.slot_values_as_set(slot_name)))
                        print(' ')
                    '''
        return stats_collection



def mean_f1(stats_dict):
    l = []
    
    for key in stats_dict:
        l.append(stats_dict[key].f1())
        
    arr = np.array(l)
    return np.mean(arr)
    


slot_filling = SlotFilling()
slot_filling.prepare() 
learning_rate = 0.01
stats_collection = slot_filling.train_validation(learning_rate, 50, 50)

if SETTING_PIPELINE:
    PIPELINE_STRING = '_pipeline'
else:
    PIPELINE_STRING = ''

# save stats
filename = 'stat_collections/' + SETTING_DISEASE + '_' + SETTING_CONTEXT + PIPELINE_STRING + '.stats'
f = open(filename, 'wb')
pickle.dump(stats_collection, f)
f.close()

    

    
    
''' 
# entities
stats = slot_filling.evalute_entities()
print(mean_f1(stats))
slot_filling.train_epoch(slot_filling.optimizer_entities, 0.01)    
'''