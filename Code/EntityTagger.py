import os, sys
from os.path import join, isfile
import tensorflow as tf
import tensorflow_hub as hub
from official.nlp.bert import tokenization
from official.nlp import optimization
import numpy as np
import copy

from ctro import *
from Document import *
from DocumentChunking import *
from DocumentEncoder import *
from SlotFillingModel import *
from EntityDecoder import *
from EntityAligner import *
from functions import *


MIN_SLOT_FREQ = 20
MAX_CHUNK_SIZE = 512
BERT_MODEL_DIM = 512

EPOCHS = 30


glaucoma_dump_file_path = 'glaucoma_slot_filling.dump'
diabetes_dump_file_path = 'diabetes_slot_filling.dump'
bert_model_name = "https://tfhub.dev/google/experts/bert/pubmed/2"


# train step signature
train_step_signature = [
tf.TensorSpec(shape=(None, None), dtype=tf.int32), # token ids
tf.TensorSpec(shape=(None, None), dtype=tf.int32), # sentence tokens 
tf.TensorSpec(shape=(None, None, None), dtype=tf.float32), # entity start positions
tf.TensorSpec(shape=(None, None, None), dtype=tf.float32), # entity end positions
]

forward_pass_signature = [
tf.TensorSpec(shape=(None, None), dtype=tf.int32), # token ids
tf.TensorSpec(shape=(None, None), dtype=tf.int32), # sentence tokens 
]



def compute_micro_stats(stats_dict):
    stat_result = F1Statistics()
    
    for class_name in stats_dict:
        if class_name == 'type':
            continue
        
        stat_result.true_positives += stats_dict[class_name].true_positives
        stat_result.false_positives += stats_dict[class_name].false_positives
        stat_result.num_occurences += stats_dict[class_name].num_occurences
        
    return stat_result




def print_stats_dict_f1s(stats_dict_glaucoma, stats_dict_diabetes, used_labels):
    labels = stats_dict_glaucoma.keys() | stats_dict_diabetes.keys()
    labels = labels & used_labels
    
    for label in sorted(labels):
        if label not in used_labels:
            continue
        
        line = label.replace('has', '')
        
        if label in stats_dict_glaucoma:
            line += " & {:2.2f} & {:2.2f} & {:2.2f}".format(stats_dict_glaucoma[label].precision(), stats_dict_glaucoma[label].recall(), stats_dict_glaucoma[label].f1())
        else:
            line += ' & - & - & - & '
            
        if label in stats_dict_diabetes:
            line += " & {:2.2f} & {:2.2f} & {:2.2f} \\\\ ".format(stats_dict_diabetes[label].precision(), stats_dict_diabetes[label].recall(), stats_dict_diabetes[label].f1())
        else:
            line += ' & - & - & - \\\\'
            
        print(line)
            
    # micro average
    mirco_stats_glaucoma = compute_micro_stats(stats_dict_glaucoma)
    mirco_stats_diabetes = compute_micro_stats(stats_dict_diabetes)
    
    line = 'micro average:'
    line += " & {:2.2f} & {:2.2f} & {:2.2f}".format(mirco_stats_glaucoma.precision(), mirco_stats_glaucoma.recall(), mirco_stats_glaucoma.f1())    
    line += " & {:2.2f} & {:2.2f} & {:2.2f} \\\\ ".format(mirco_stats_diabetes.precision(), mirco_stats_diabetes.recall(), mirco_stats_diabetes.f1())
    print(line)
    
    

# create bert tokenizer
FullTokenizer = tokenization.FullTokenizer
bert_layer = hub.KerasLayer(bert_model_name, trainable=False) 
vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy() # The vocab file of bert for tokenizer
tokenizer = FullTokenizer(vocab_file)  


def convert_abstracts_to_documents(abstracts, tokenizer):
    documents = []
    
    for abstract in abstracts:
        doc = Document()
        doc.set_from_abstract(abstract, tokenizer)
        documents.append(doc)
        
    return documents



def create_document_chunkings(documents, max_chunk_size):
    documnet_chunkings = []
    
    for doc in documents:
        doc_chunking = DocumentChunking(max_chunk_size)
        doc_chunking.set_from_document(doc)
        documnet_chunkings.append(doc_chunking)
        
    return documnet_chunkings



def create_batches(document_chunkings, document_encoder, slot_indices):
    batches = []
    
    for doc_chunking in document_chunkings:
        batch = dict()
        num_slot_labels = len(slot_indices)
        
        # tokens
        batch['token_ids'] = document_encoder.encode_tokens_document_chunking(doc_chunking)
        batch['token_masks'] = document_encoder.create_token_masks_document_chunking(doc_chunking)
        
        # entities
        encoded_start_positions, encoded_end_positions = document_encoder.encode_entity_positions_document_chunking(doc_chunking)
        batch['entity_start_positions'] = tf.one_hot(encoded_start_positions, num_slot_labels)
        batch['entity_end_positions'] = tf.one_hot(encoded_end_positions, num_slot_labels)
        
        batches.append(batch)
        
    return batches

        


class EntityTagger:
    def import_documents(self, dump_file_path):
        f = open(dump_file_path, 'rb')
        abstracts_train, abstracts_validation, abstracts_test = pickle.load(f, encoding='latin1')
        f.close()
        
        self.documents_train = convert_abstracts_to_documents(abstracts_train, tokenizer)
        self.documents_validation = convert_abstracts_to_documents(abstracts_validation, tokenizer)
        self.documents_test = convert_abstracts_to_documents(abstracts_test, tokenizer)
        
        
        
    def count_slots(self):
        slot_counts = dict()
        
        for document in self.documents_train:
            for sentence in document.get_sentences():
                for entity in sentence.get_entities():
                    
                    for slot_name in list(entity.get_referencing_slot_names()):

                        if slot_name in slot_counts:
                            slot_counts[slot_name] += 1
                        else:
                            slot_counts[slot_name] = 0
                    
        return slot_counts
    
    
    
    def create_slot_indices(self, min_slot_freq):
        slot_counts = self.count_slots()
        slot_indices = dict()
        slot_indices_reverse = dict()
        
        # index of 'no_slot' label
        slot_indices['no_slot'] = 0
        slot_indices_reverse[0] = 'no_slot'
        
        # 'real' slot labels
        for slot_name in slot_counts:
            
            if slot_counts[slot_name] >= min_slot_freq:
                slot_index = len(slot_indices)
                
                slot_indices[slot_name] = len(slot_indices)
                slot_indices_reverse[slot_index] = slot_name
                
        # save dicts as object variables
        self.slot_indices = slot_indices
        self.slot_indices_reverse = slot_indices_reverse
        self.used_labels = set(slot_indices.keys())
        
        
        
    def create_document_chunkings(self, max_chunk_size):
        self.document_chunkings_train = create_document_chunkings(self.documents_train, max_chunk_size)
        self.document_chunkings_validation = create_document_chunkings(self.documents_validation, max_chunk_size)
        self.document_chunkings_test = create_document_chunkings(self.documents_test, max_chunk_size)
        
        
        
    def create_batches(self):
        # create BERT tokenizer
        FullTokenizer = tokenization.FullTokenizer
        bert_layer = hub.KerasLayer(bert_model_name, trainable=False) 
        vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
        tokenizer = FullTokenizer(vocab_file)

        # create document encoder
        document_encoder = DocumentEncoder(tokenizer, self.slot_indices, 'no_slot')
        
        # create batches
        self.batches_train = create_batches(self.document_chunkings_train, document_encoder, self.slot_indices)
        self.batches_validation = create_batches(self.document_chunkings_validation, document_encoder, self.slot_indices)
        self.batches_test = create_batches(self.document_chunkings_test, document_encoder, self.slot_indices)
        
        
        
    def create_optimizer(self):
        steps_per_epoch = len(self.batches_train)
        num_train_steps = steps_per_epoch * EPOCHS
        num_warmup_steps = int(0.1*num_train_steps)
        
        init_lr = 3e-5
        self.optimizer = optimization.create_optimizer(init_lr=init_lr,
                                                  num_train_steps=num_train_steps,
                                                  num_warmup_steps=num_warmup_steps,
                                                  optimizer_type='adamw')
        
        
    def create_model(self):
        self.slot_filling_model = SlotFillingModel(bert_model_name, BERT_MODEL_DIM, len(self.slot_indices))
        
        
        
    def prepare(self):
        self.create_slot_indices(MIN_SLOT_FREQ)
        self.create_document_chunkings(MAX_CHUNK_SIZE)
        self.create_batches()
        self.create_model()
        self.create_optimizer()
        
        # keras metrics
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        
        
        
    def call_bert_layer(self, token_ids, token_ids_masks, training=False):
        input_dict = {}
        input_dict['input_word_ids'] = token_ids
        input_dict['input_mask'] = token_ids_masks
        input_dict['input_type_ids'] = tf.zeros_like(token_ids, dtype=tf.int32)
        
        return self.slot_filling_model.bert_layer(input_dict, training=training)["sequence_output"]
        
        
        
    @tf.function(input_signature=train_step_signature)
    def train_step(self, token_ids, token_id_masks, entity_start_position_labels, entity_end_position_labels):
        with tf.GradientTape() as tape:
            token_embeddings = self.call_bert_layer(token_ids, token_id_masks, training=True)
            
            start_position_logits = self.slot_filling_model.dense_entity_start_positions(token_embeddings)
            end_position_logits = self.slot_filling_model.dense_entity_end_positions(token_embeddings)
            
            loss_entity_start_positions = tf.nn.softmax_cross_entropy_with_logits(labels=entity_start_position_labels, logits=start_position_logits)
            loss_entity_end_positions = tf.nn.softmax_cross_entropy_with_logits(labels=entity_end_position_labels, logits=end_position_logits)
            
            # mask invalid positions
            bool_token_id_masks = tf.greater(token_id_masks, 0)
            num_tokens = tf.reduce_sum(token_id_masks)
            num_tokens = tf.cast(num_tokens, tf.float32)
            
            loss_entity_start_positions = tf.boolean_mask(loss_entity_start_positions, bool_token_id_masks)
            loss_entity_end_positions = tf.boolean_mask(loss_entity_end_positions, bool_token_id_masks)
            
            loss_entity_start_positions = tf.reduce_sum(loss_entity_start_positions) / num_tokens
            loss_entity_end_positions = tf.reduce_sum(loss_entity_end_positions) / num_tokens
            
            # total loss
            loss = loss_entity_start_positions + loss_entity_end_positions
    
        # gradient descent
        trainable_variables = self.slot_filling_model.trainable_weights
        gradients = tape.gradient(loss, trainable_variables)
        #gradients = [tf.clip_by_norm(gradient, 1.0) for gradient in gradients]
        self.optimizer.apply_gradients(zip(gradients, trainable_variables))
        
        # update metrics
        self.train_loss(loss)
        
        
        
    def train(self, num_epochs):
        for i in range(num_epochs):
            for batch in self.batches_train:
                self.train_step(batch['token_ids'],
                                batch['token_masks'],
                                batch['entity_start_positions'],
                                batch['entity_end_positions'])
            
            # print loss after each epoch
            print('loss: ' + str(self.train_loss.result()))
            self.train_loss.reset_states()
            
            
            
    def save_model_weights(self, filename):
        self.slot_filling_model.save_weights(filename)
        
        
        
    def load_model_weights(self, filename):
        self.slot_filling_model.load_weights(filename)
        
        
        
    @tf.function(input_signature=forward_pass_signature)
    def forward_pass_entity_positions(self, token_ids, token_id_masks):
        token_embeddings = self.call_bert_layer(token_ids, token_id_masks, training=False)
        
        entity_start_positions = self.slot_filling_model.dense_entity_start_positions(token_embeddings)
        entity_end_positions = self.slot_filling_model.dense_entity_end_positions(token_embeddings)
        
        return entity_start_positions, entity_end_positions
        
        
        
    def evaluate_entity_prediction(self):
        entity_decoder = EntityDecoder(self.slot_indices, 'no_slot')
        entity_aligner = EntityAligner()
        stats_dict = dict()
        
        for i in range(len(self.documents_test)):
            batch = self.batches_test[i]
            document = self.documents_test[i]
            doc_chunking = self.document_chunkings_test[i]
            
            # predict entity positions
            entity_start_positions, entity_end_positions = self.forward_pass_entity_positions(batch['token_ids'],
                                                                                              batch['token_masks'])
            
            entity_start_positions = tf.argmax(entity_start_positions, axis=-1)
            entity_end_positions = tf.argmax(entity_end_positions, axis=-1)
            
            # get gt slot referenced entities
            gt_entities = [entity for entity in document.get_entities() if len(entity.get_referencing_slot_names()) > 0]
            
            # predict entities
            predicted_entities = entity_decoder.decode_document_chunking(entity_start_positions,
                                                                         entity_end_positions, 
                                                                         doc_chunking)
            
            document.set_entity_tokens(predicted_entities)
            
            for entity in predicted_entities:
                slot_names = list(entity.get_referencing_slot_names())
                if len(slot_names) == 0:
                    continue
                
                slot_name = slot_names[0]
                
                if slot_name == 'hasEndoPointDescription':
                    print(' '.join(entity.get_tokens()).replace(' ##', ''))
            
            aligned_entity_pairs = entity_aligner.align_entities_exact(gt_entities, predicted_entities)
            entity_aligner.update_stats_dict(stats_dict, aligned_entity_pairs)
            
        #print_stats_dict_f1s(stats_dict, stats_dict, self.used_labels)
        return stats_dict
    
    
    
    def get_test_set_document(self, pubmed_id):
        for document in self.documents_test:
            abstract = document.get_abstract()
            
            if abstract.abstract_id == pubmed_id:
                return document
            
        raise Exception('Abstract not found')
    
    
    
    def predict_document_entities(self, document, document_encoder, entity_decoder, max_chunk_size=MAX_CHUNK_SIZE):
        # create document chunking of document
        doc_chunking = DocumentChunking(max_chunk_size)
        doc_chunking.set_from_document(document)
        
        # encode tokens
        token_ids = document_encoder.encode_tokens_document_chunking(doc_chunking)
        token_mask = document_encoder.create_token_masks_document_chunking(doc_chunking)
        
        # predict entity positions
        entity_start_positions, entity_end_positions = self.forward_pass_entity_positions(token_ids, token_mask)
        entity_start_positions = tf.argmax(entity_start_positions, axis=-1)
        entity_end_positions = tf.argmax(entity_end_positions, axis=-1)

        # predict entities
        predicted_entities = entity_decoder.decode_document_chunking(entity_start_positions,
                                                                         entity_end_positions, 
                                                                         doc_chunking)
            
        document.set_entity_tokens(predicted_entities)
        return predicted_entities
    
    
    
    def update_covergae_stats_dict(self, gt_entities, predicted_entities, stats_dict, used_slots):
        for slot_name in used_slots:
            if slot_name == 'type':
                continue
            if slot_name not in stats_dict:
                stats_dict[slot_name] = F1Statistics()
                
        # update num_occurences count
        for entity in gt_entities:
            referencing_slot_names = list(entity.get_referencing_slot_names())
                
            if len(referencing_slot_names) == 0:
                continue
                
            slot_name = referencing_slot_names[0]
            
            if slot_name == 'type' or slot_name not in used_slots:
                continue
                
            stats_dict[slot_name].num_occurences += 1
            
        # update true positives, false positives
        for slot_name in used_slots:
            if slot_name == 'type':
                continue
            
            selected_gt_entities = select_entities_by_referencing_slot(gt_entities, slot_name)
            selected_predicted_entities = select_entities_by_referencing_slot(predicted_entities, slot_name)
            
            gt_entities_strings = {' '.join(entity.get_tokens()) for entity in selected_gt_entities}
            predicted_entities_strings = {' '.join(entity.get_tokens()) for entity in selected_predicted_entities}
            
            stats = stats_dict[slot_name]
            stats.true_positives += len(gt_entities_strings & predicted_entities_strings)
            stats.false_positives += len(predicted_entities_strings - gt_entities_strings)
            
            
            
    def update_covergae_stats_dict_partial_old(self, gt_entities, predicted_entities, stats_dict, used_slots):
        for slot_name in used_slots:
            if slot_name == 'type':
                continue
            if slot_name not in stats_dict:
                stats_dict[slot_name] = F1Statistics()
                
        # update num_occurences count
        for entity in gt_entities:
            referencing_slot_names = list(entity.get_referencing_slot_names())
                
            if len(referencing_slot_names) == 0:
                continue
                
            slot_name = referencing_slot_names[0]
            
            if slot_name == 'type' or slot_name not in used_slots:
                continue
                
            stats_dict[slot_name].num_occurences += 1
            
        # update true positives, false positives
        for slot_name in used_slots:
            if slot_name == 'type':
                continue
            
            selected_gt_entities = select_entities_by_referencing_slot(gt_entities, slot_name)
            selected_predicted_entities = select_entities_by_referencing_slot(predicted_entities, slot_name)
            
            gt_entities_token_sets = [set(entity.get_tokens()) for entity in selected_gt_entities]
            predicted_entities_token_sets = [set(entity.get_tokens()) for entity in selected_predicted_entities]
            
            stat = stats_dict[slot_name]
            for predicted_entity in predicted_entities_token_sets:
                found = False
                for gt_entity in gt_entities_token_sets:
                    if len(predicted_entity & gt_entity) > 0:
                        print(gt_entity, predicted_entity, predicted_entity & gt_entity)
                        print('....')
                        found = True
                        gt_entity_remove = gt_entity
                        break
                    
                if found:
                    stat.true_positives += 1
                    gt_entities_token_sets.remove(gt_entity_remove)
                else:
                    stat.false_positives += 1
                    
        return stats_dict
    
    
    
    def evaluate_full_text_coverage(self, directory, disease_prefix, used_slots):
        document_encoder = DocumentEncoder(tokenizer, self.slot_indices, 'no_slot')
        entity_decoder = EntityDecoder(self.slot_indices, 'no_slot')
        stats_dict = dict()
        
        filenames = [f for f in os.listdir(directory) if isfile(join(directory, f)) and disease_prefix in f]
        
        # process files
        for filename in filenames:
            # create document for full text article
            path = join(directory, filename)
            full_text_document = create_document_from_file(path, tokenizer)
            
            # get corresponding document in test set
            pubmed_id = filename.replace(disease_prefix, '')
            pubmed_id = pubmed_id.replace('.txt', '')
            gt_document = self.get_test_set_document(pubmed_id)
            
            gt_entities = gt_document.get_entities()
            predicted_entities = self.predict_document_entities(full_text_document, document_encoder, entity_decoder)
            
            self.update_covergae_stats_dict(gt_entities, predicted_entities, stats_dict, used_slots)
            
        return stats_dict
    
    
    
    def evaluate_full_text_coverage_partial(self, directory, disease_prefix, used_slots):
        document_encoder = DocumentEncoder(tokenizer, self.slot_indices, 'no_slot')
        entity_decoder = EntityDecoder(self.slot_indices, 'no_slot')
        stats_dict = dict()
        
        filenames = [f for f in os.listdir(directory) if isfile(join(directory, f)) and disease_prefix in f]
        
        # process files
        for filename in filenames:
            # create document for full text article
            path = join(directory, filename)
            full_text_document = create_document_from_file(path, tokenizer)
            
            # get corresponding document in test set
            pubmed_id = filename.replace(disease_prefix, '')
            pubmed_id = pubmed_id.replace('.txt', '')
            gt_document = self.get_test_set_document(pubmed_id)
            
            gt_entities = gt_document.get_entities()
            predicted_entities = self.predict_document_entities(full_text_document, document_encoder, entity_decoder)
            
            self.update_covergae_stats_dict_partial(gt_entities, predicted_entities, stats_dict, used_slots)
            
        return stats_dict
            
            
        
    

       

print('Glaucoma ===============================')        
entity_tagger_glaucoma = EntityTagger()
entity_tagger_glaucoma.import_documents(glaucoma_dump_file_path)
entity_tagger_glaucoma.prepare()
entity_tagger_glaucoma.load_model_weights('model_ner_glaucoma')
#stats_dict_glaucoma = entity_tagger_glaucoma.evaluate_entity_prediction()


print('T2DM ===============================')  
entity_tagger_diabetes = EntityTagger()
entity_tagger_diabetes.import_documents(diabetes_dump_file_path)
entity_tagger_diabetes.prepare()
entity_tagger_diabetes.load_model_weights('model_ner_diabetes')
#stats_dict_diabetes = entity_tagger_diabetes.evaluate_entity_prediction()


used_slots = entity_tagger_glaucoma.used_labels | entity_tagger_diabetes.used_labels
used_slots.remove('no_slot')
#print_stats_dict_f1s(stats_dict_glaucoma, stats_dict_diabetes, used_slots)
#sys.exit()

# coverage ###########################################################################
coverage_stats_dict_gl = entity_tagger_glaucoma.evaluate_full_text_coverage('full_texts', 'gl_', used_slots)
coverage_stats_dict_dm2 = entity_tagger_diabetes.evaluate_full_text_coverage('full_texts', 'dm2_', used_slots)
print_stats_dict_f1s(coverage_stats_dict_gl, coverage_stats_dict_dm2, used_slots)


'''
#entity_tagger.load_model_weights('model_ner_diabetes')
entity_tagger.train(EPOCHS)
entity_tagger.save_model_weights('model_ner_diabetes')
entity_tagger.evaluate_entity_prediction()
'''



    
    
    