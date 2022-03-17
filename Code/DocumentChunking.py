from Chunk import *
from Document import *
        
        


class DocumentChunking:
    def __init__(self, max_chunk_size):   
        # max tokens of each chunk
        self._max_chunk_size = max_chunk_size
        
        # list of chunks
        self._chunks = []
        
        
    def append_chunk(self, chunk):
        self._chunks.append(chunk)
            
            
            
    def set_from_document(self, document):
        self._chunks = []
        
        # split sentences of document into blocks which
        # fit into chunks
        sentence_blocks = document.split_sentence_seq_for_chunking(self.get_max_chunk_size())
        
        # create chunks from sentence blocks
        current_chunk_index = 0
        
        for sentence_block in sentence_blocks:
            chunk = Chunk(sentence_block, current_chunk_index)
            self.append_chunk(chunk)
            current_chunk_index += 1

        
    
    def get_max_chunk_size(self):
        return self._max_chunk_size
    
    
    def get_num_chunks(self):
        return len(self._chunks)
    
    
    def get_chunks(self):
        return self._chunks
    
    
    def get_chunk_by_index(self, chunk_index):
        chunks_dict = {chunk:chunk.get_chunk_index() for chunk in self.get_chunks()}
        
        # check if chunk index is valid
        if chunk_index not in chunks_dict:
            raise IndexError('Invalid chunk index')
            
        return chunks_dict[chunk_index]
    
    
    def append_chunk(self, chunk):
        self._chunks.append(chunk)
        
        
    def set_entity_chunk_indices(self, entities):
        for entity in entities:
            chunk_found = False
            entity_sentence_index = entity.get_sentence_index()
            
            for chunk in self.get_chunks():
                if entity_sentence_index in chunk.get_sentence_indices():
                    chunk_found = True
                    entity.set_chunk_index(chunk.get_chunk_index())
                    break
                
            # check if a chunk for current entity was found
            if not chunk_found:
                raise Exception('ERROR: no chunk found for entity')
                
                
                
    def get_entity_start_end_indices(self, entities):
        # lists of [chunk_index, in_chunk_offset] lists representing indices
        # into doc chunking of entity start/end positions
        entity_start_indices = []
        entity_end_indices = []
        
        for entity in entities:
            chunk_index = None
            entity_sentence_index = entity.get_sentence_index()
            
            # get chunk containing entity ################################
            for chunk in self.get_chunks():
                if entity_sentence_index in chunk.get_sentence_indices():
                    chunk_index = chunk.get_chunk_index()
                    break
                
            # check if a chunk for current entity was found
            if chunk_index is None:
                raise Exception('ERROR: no chunk found for entity')
                
            # compute start/end positions of entity in chunk
            in_chunk_sentence_offset = chunk.get_sentence_offset(entity_sentence_index)
            in_chunk_start_pos = in_chunk_sentence_offset + entity.get_start_pos()
            in_chunk_end_pos = in_chunk_sentence_offset + entity.get_end_pos()
            
            # add computed indices to list
            entity_start_indices.append([chunk_index, in_chunk_start_pos])
            entity_end_indices.append([chunk_index, in_chunk_end_pos])
            
        return entity_start_indices, entity_end_indices
    
    
    
    
    
    