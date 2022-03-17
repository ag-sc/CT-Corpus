import copy


class Entity:
    def __init__(self, entity_index=None, sentence_index=None, start_pos=None,
                 end_pos=None, label=None, tokens=None):
        # list of tokens representing entity
        self._tokens = tokens
        
        # global document-level entity index, 0-based
        self._global_entity_index = entity_index
        
        # index of sentence  which contains entity; space of sentence indices is per document, 0-based
        self._sentence_index = sentence_index
        
        # in sentence token offset of first entity token
        self._start_pos = start_pos
        
        # in sentence token offset of last entity token
        self._end_pos = end_pos
        
        # entity class label (e.g. most general superclass)
        self._label = label
        
        # set of slot names by which entity is referenced, if any
        self._referencing_slot_names = set()
        
        # set of template ids indicating referencing templates
        self._referencing_template_ids = set()
        
        # vector representation of entity
        self._vector_representation = None
        
        # chunk index when DocumentChunking is used
        self._chunk_index = None
        
        
    def copy(self):
        return copy.copy(self)
    
    
    def __eq__(self, other):
        if not isinstance(other, Entity):
            raise TypeError()
            
        return self.__dict__ == other.__dict__
        
        
    def __neq__(self, other):
        return not self == other
    
    
    def __hash__(self):
        if self.get_global_entity_index() is None:
            raise Exception('Hash of entity without index is undefined')
            
        return self.get_global_entity_index()
        
    
    def set_tokens(self, tokens):
        self._tokens = []
        
        for token in tokens:
            self._tokens.append(token)
            
            
    def get_tokens(self):
        return self._tokens
    
    
    def set_global_entity_index(self, global_entity_index):
        self._global_entity_index = global_entity_index
        
        
    def get_global_entity_index(self):
        return self._global_entity_index
    
    
    def set_sentence_index(self, sentence_index):
        self._sentence_index = sentence_index
        
        
    def get_sentence_index(self):
        return self._sentence_index
    
    
    def get_start_pos(self):
        return self._start_pos
    
    
    def set_start_pos(self, start_pos):
        self._start_pos = start_pos
        
        
    def get_end_pos(self):
        return self._end_pos
    
    
    def set_end_pos(self, end_pos):
        self._end_pos = end_pos
        
        
    def set_label(self, label):
        self._label = label
        
        
    def get_label(self):
        return self._label
    
        
    def get_referencing_slot_names(self):
        return self._referencing_slot_names
    
    
    def add_referencing_slot_name(self, slot_name):
        self._referencing_slot_names.add(slot_name)
        
        
    def get_referencing_template_ids(self):
        return self._referencing_template_ids
    
    
    def add_referencing_template_ids(self, template_id):
        self._referencing_template_ids.add(template_id)
        
        
    def get_vector_representation(self):
        return self._vector_representation
        
        
    def set_vector_representation(self, vector):
        self._vector_representation = vector
        
        
    def get_chunk_index(self):
        return self._chunk_index
    
    
    def set_chunk_index(self, chunk_index):
        self._chunk_index = chunk_index
        
        
    def print_out(self):
        print('--- Entity start ---')
        
        print('sentence index:', self.get_sentence_index())
        print('global entity index:', self.get_global_entity_index())
        print('sentence index:', self.get_sentence_index())
        print('chunk index:', self.get_chunk_index())
        print('entity tokens:', self.get_tokens())
        print('start pos:', self.get_start_pos())
        print('end pos:', self.get_end_pos())
        print('entity label:', self.get_label())
        print('referencing slots:', self.get_referencing_slot_names())
        print('referencing templates:', self.get_referencing_template_ids())
        
        print('--- Entity end ---')
    
    
    
    
    
    
    