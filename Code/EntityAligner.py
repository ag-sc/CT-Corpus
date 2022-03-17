from EntityCollection import *
from ctro import *



class AlignedEntityPair:
    def __init__(self):
        self.ground_truth_entity = None
        self.predicted_entity = None
        
        
    def print_out(self):
        print('---- gt entity ------')
        self.ground_truth_entity.print_out()
        
        print('---- predicted entity ------')
        self.predicted_entity.print_out()
        
        print('==============================')
        
        
        
        
class EntityAligner:
    def __init__(self, use_slots=True, neutral_label='no_slot'):
        self._use_slots = use_slots
        self._neutral_label = neutral_label
        
        
    def find_exact_matching_predicted_entity(self, gt_entity, predicted_entities):
        if self._use_slots:
            gt_label = list(gt_entity.get_referencing_slot_names())[0]
        else:
            gt_label = gt_entity.get_label()
            
        aligned_entity_pair = AlignedEntityPair()
        aligned_entity_pair.ground_truth_entity = gt_entity
         
            
        for predicted_entity in predicted_entities:
            # entities have to be from same sentence
            if gt_entity.get_sentence_index() != predicted_entity.get_sentence_index():
                continue
            
            if self._use_slots:
                predicted_label = list(predicted_entity.get_referencing_slot_names())[0]
            else:
                predicted_label = predicted_entity.get_label()
                
            if gt_label == predicted_label:
                if gt_entity.get_start_pos() == predicted_entity.get_start_pos():
                    if gt_entity.get_end_pos() == predicted_entity.get_end_pos():
                        aligned_entity_pair.predicted_entity = predicted_entity
                        return aligned_entity_pair
                    
        # no matching predicted entity was found
        return aligned_entity_pair
        
        
        
    def align_entities_exact(self, ground_truth_entities, predicted_entities):
        aligned_entity_pairs = []
        
        while len(ground_truth_entities) > 0:
            gt_entity = ground_truth_entities[0]
            aligned_entity_pair = self.find_exact_matching_predicted_entity(gt_entity, predicted_entities)
            
            ground_truth_entities.remove(gt_entity)
            if aligned_entity_pair.predicted_entity is not None:
                if aligned_entity_pair.predicted_entity in predicted_entities:
                    predicted_entities.remove(aligned_entity_pair.predicted_entity)
                
            aligned_entity_pairs.append(aligned_entity_pair)
            
        # not assigned predicted entities
        for predicted_entity in predicted_entities:
            aligned_entity_pair = AlignedEntityPair()
            aligned_entity_pair.predicted_entity = predicted_entity
            aligned_entity_pairs.append(aligned_entity_pair)
            
        return aligned_entity_pairs
    
    
    
    def update_stats_dict(self, stats_dict, aligned_entity_pairs):
         for aligned_entity_pair in aligned_entity_pairs:
            gt_label = None
            predicted_label = None
            
            if self._use_slots:
                if aligned_entity_pair.ground_truth_entity is not None:
                    gt_label = list(aligned_entity_pair.ground_truth_entity.get_referencing_slot_names())[0]
                if aligned_entity_pair.predicted_entity is not None:
                    predicted_label =  list(aligned_entity_pair.predicted_entity.get_referencing_slot_names())[0]
            else:
                if aligned_entity_pair.ground_truth_entity is not None:
                    gt_label = aligned_entity_pair.ground_truth_entity.get_label()
                if aligned_entity_pair.predicted_entity is not None:
                    predicted_label = aligned_entity_pair.predicted_entity.get_label()
                    
            if gt_label == 'type' or predicted_label == 'type':
                continue
                
            if gt_label is not None and gt_label not in stats_dict:
                stats_dict[gt_label] = F1Statistics()
            if predicted_label is not None and predicted_label not in stats_dict:
                stats_dict[predicted_label] = F1Statistics()
            
            if gt_label is not None:
                stats_dict[gt_label].num_occurences += 1
            else:
                stats_dict[predicted_label].false_positives += 1
            
            if gt_label is not None and predicted_label is not None:
                if predicted_label != self._neutral_label:
                    if predicted_label == gt_label:
                        stats_dict[predicted_label].true_positives += 1
                    else:
                        stats_dict[predicted_label].false_positives += 1
                    
         return stats_dict
            
    
    
    
    
    
    
    