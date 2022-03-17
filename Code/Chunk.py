import numpy as np



class Chunk:
    def __init__(self, sentences, chunk_index):
        self._sentences = list(sentences)
        
        # chunk index within DocumentChunking
        self._chunk_index = chunk_index
        
        # token offsets of sentences in chunk
        # key: sentence index
        # value: token offset of sentence in chunk
        self._sentence_offsets = dict()
        
        
        # set sentence offsets
        current_sentence_offset = 1 # [CLS] token is first token in each chunk
        
        for sentence in self.get_sentences():
            # get index of sentence
            sentence_index = sentence.get_index()
            assert sentence_index is not None
            
            # srt in chunk offset of sentence
            self._sentence_offsets[sentence_index] = current_sentence_offset
            current_sentence_offset += sentence.get_num_tokens() + 1 # + 1 because of [SEP] token
            
        
        
    def get_sentences(self):
        return self._sentences
    
    
    def get_num_sentences(self):
        return len(self._sentences)
    
    
    def set_chunk_index(self, chunk_index):
        self._chunk_index = chunk_index
        
        
    def get_chunk_index(self):
        return self._chunk_index
    
    
    def get_sentence_indices(self):
        return {sentence.get_index() for sentence in self.get_sentences()}
    
    
    
    def get_sentence_by_index(self, sentence_index):
        if sentence_index not in self._sentence_offsets:
            raise IndexError('Invalid sentence index')
            
        for sentence in self.get_sentences():
            if sentence.get_index() == sentence_index:
                return sentence
            
        # sentence was not found
        raise Exception('Sentence not found in chunk')
    
    
    
    def get_sentence_offset(self, sentence_index):
        if sentence_index not in self._sentence_offsets:
            raise IndexError('Invalid sentence index')
            
        return self._sentence_offsets[sentence_index]
    
    
    
    def extract_sentence_subarray(self, chunk_array, sentence_index):
        # estimate sentence boundaries
        sentence = self.get_sentence_by_index(sentence_index)
        sentence_start_offset = self.get_sentence_offset(sentence_index)
        sentence_end_offset = sentence_start_offset + sentence.get_num_tokens()

        # return numpy subarray of chunk
        return chunk_array[sentence_start_offset:sentence_end_offset]
    
    
    
    def set_entity_chunk_indices(self):
        chunk_index = self
    

        
        
