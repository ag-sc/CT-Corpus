from ctro import *
from Sentence import *
from Document import *


def select_entities_by_template_type(entities, template_type):
    template_slots = ontology.group_slots[template_type]
    
    return [entity for entity in entities if len(template_slots & entity.get_referencing_slot_names()) > 0]



def select_entities_by_referencing_slot(entities, slot_name):
    return [entity for entity in entities if len(set([slot_name]) & entity.get_referencing_slot_names()) > 0]



def print_group(group, slots):
    slots = sorted(slots)
    
    for slot_name in slots:
        if slot_name in group.slots:
            slot_fillers = []
            
            for slot_value in group.slots[slot_name]:
                slot_fillers.append(slot_value.string.replace(' ##', ''))
                
            print(slot_name + ': ' + ' | '.join(slot_fillers))
        


def print_groups_dict(groups_dict, used_slots):
    for group_name in ontology.used_group_names:
        if group_name in groups_dict:
            groups = groups_dict[group_name]
            print('Template Type: ', group_name, ' ===========================')
            for i,group in enumerate(groups):
                print('Instance ', i+1)
                print_group(group, used_slots)
                print()
                print('-----------')
                
                
                
def create_document_from_file(path, tokenizer):
    sentences = []
    
    # read sentences
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            tokens = tokenizer.tokenize(line)
            sentence = Sentence(tokens)
            sentences.append(sentence)
            
    # assign unique index to each sentence
    for i,sentence in enumerate(sentences):
        sentence.set_index(i)
        
    # create and return final document
    return Document(sentences)
        
        
        