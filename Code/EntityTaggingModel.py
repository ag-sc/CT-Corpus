import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.layers import Embedding,Conv1D,Dense,Dropout



class EntityTaggingModel(tf.keras.Model):
    def __init__(self, bert_model_name, bert_model_dim, num_entity_labels):
        super(SlotFillingModel, self).__init__()
        
        self.bert_layer = hub.KerasLayer(bert_model_name, trainable=True)
        self.dense_entity_start_positions = Dense(num_entity_labels)
        self.dense_entity_end_positions = Dense(num_entity_labels)