import sys, os, itertools, numpy as np
from ctro import *


def import_duplicated_trials_ids(filename):
    l = []
    f = open(filename)
    
    for line in f:
        line = line.strip()
        l.append(line)
        
    return l



def create_filename_pairs(ids, file_ending):
    filenames = os.listdir('data3/')
    pairs = []
    
    for i in ids:
        pair = []
        
        for filename in filenames:
            if 'csv' not in file_ending:
                if i in filename and file_ending in filename and 'admin' in filename:
                    pair.append(filename)
            else:
                if i in filename and file_ending in filename:
                    pair.append(filename)
        pairs.append(pair)
        
    return pairs



def sort_filename_pairs(pairs, b_copy_predicted):
    result = []
    
    for a,b in pairs:
        if b_copy_predicted:
            if 'copy' in a:
                result.append((b,a))
            else:
                result.append((a,b))
        else:
            if 'copy' in a:
                result.append((a,b))
            else:
                result.append((b,a))
                
    return result



def pop_max_key(d):
    m = -1
    max_key = None
    
    for key in d:
        if d[key] > m:
            m = d[key]
            max_key = key
    
    del d[max_key]        
    return max_key



def update_agreement_statistics(statistics, group_dict_gt, group_dict_predicted):
    ids_gt = set(group_dict_gt.keys())
    ids_predicted = set(group_dict_predicted.keys())
    scores = compute_group_pair_scores(group_dict_gt, group_dict_predicted)
    
    while len(ids_gt) > 0 and len(ids_predicted) > 0:
        id_gt,id_predicted = pop_max_key(scores)
        if id_gt not in ids_gt or id_predicted not in ids_predicted:
            continue
        
        statistics.update(group_dict_gt[id_gt], group_dict_predicted[id_predicted])
        
        ids_gt.remove(id_gt)
        ids_predicted.remove(id_predicted)
        
    for id_gt in ids_gt:
        statistics.update_num_occurences(group_dict_gt[id_gt])
    for id_predicted in ids_predicted:
        statistics.update_false_positives(group_dict_predicted[id_predicted])
        
        
        
def compute_best_alignment(group_collection_gt, group_collection_predicted):
    alignment = {}
    
    for group_name in group_collection_gt.groups:
        group_pair_scores = {}
        
        # compute scores of all group pairings
        for group_id_gt in group_collection_gt.get_group_name_ids(group_name):
            for group_id_predicted in group_collection_predicted.get_group_name_ids(group_name):
                group_gt = group_collection_gt.get_group(group_name, group_id_gt)
                group_predicted = group_collection_predicted.get_group(group_name, group_id_predicted)
                
                statistics = F1StatisticsCollection()
                statistics.update(group_gt, group_predicted)
                
                score = statistics.get_micro_stats().f1()
                group_pair_scores[(group_id_gt,group_id_predicted)] = score
                
        # estimate best alignment according to scores
        all_group_gt_ids = group_collection_gt.get_group_name_ids(group_name)
        all_group_predicted_ids = group_collection_predicted.get_group_name_ids(group_name)
        
        while len(all_group_gt_ids) > 0 and len(all_group_predicted_ids) > 0:
            id_gt,id_predicted = pop_max_key(group_pair_scores)
            
            if id_gt not in all_group_gt_ids or id_predicted not in all_group_predicted_ids:
                continue
            
            alignment[id_gt] = id_predicted
            
            all_group_gt_ids.remove(id_gt)
            all_group_predicted_ids.remove(id_predicted)
            
    return alignment
        
        

def slot_filling_agreement(filename_pairs_tokenization, filename_pairs_annodb, filename_pairs_triples):
    assert len(filename_pairs_annodb)  == len(filename_pairs_triples)
    statistics = F1StatisticsCollection()
    
    for i in range(len(filename_pairs_annodb)):
        # load data ################################################################
        tokenization_gt = Tokenization()
        tokenization_gt.set_from_file('data3/' + filename_pairs_tokenization[i][0])
        
        tokenization_predicted = Tokenization()
        tokenization_predicted.set_from_file('data3/' + filename_pairs_tokenization[i][1])
        
        annotated_abstract_gt = AnnotatedAbstract()
        annotated_abstract_gt.set_from_file('data3/' + filename_pairs_annodb[i][0], tokenization_gt)
        
        annotated_abstract_predicted = AnnotatedAbstract()
        annotated_abstract_predicted.set_from_file('data3/' + filename_pairs_annodb[i][1], tokenization_predicted)
        
        group_collection_gt = GroupCollection()
        group_collection_gt.import_group_ids('data3/' + filename_pairs_triples[i][0])
        
        group_collection_predicted = GroupCollection()
        group_collection_predicted.import_group_ids('data3/' + filename_pairs_triples[i][1])
        
        # fill slots
        group_collection_gt.fill(annotated_abstract_gt, False)
        group_collection_predicted.fill(annotated_abstract_predicted, False)
        
        # compute best alignment
        best_alignment = compute_best_alignment(group_collection_gt, group_collection_predicted)
        
        # import high level slot values
        group_collection_gt.import_high_level_slot_values('data3/' + filename_pairs_triples[i][0])
        group_collection_predicted.import_high_level_slot_values('data3/' + filename_pairs_triples[i][1])
        
        group_collection_gt.align_high_level_slot_values(best_alignment)
        
        for group_name in ontology.group_names:
            all_group_gt_ids = group_collection_gt.get_group_name_ids(group_name)
            all_group_predicted_ids = group_collection_predicted.get_group_name_ids(group_name)
            
            for id_gt in group_collection_gt.get_group_name_ids(group_name):
                if id_gt in best_alignment:
                    group_gt = group_collection_gt.get_group(group_name, id_gt)
                    group_predicted = group_collection_predicted.get_group(group_name, best_alignment[id_gt])
                    statistics.update(group_gt, group_predicted)
                    
                    all_group_gt_ids.remove(id_gt)
                    all_group_predicted_ids.remove(best_alignment[id_gt])
                    
            for id_gt in all_group_gt_ids:
                statistics.update_num_occurences(group_collection_gt.get_group(group_name, id_gt))
            for id_predicted in all_group_predicted_ids:
                statistics.update_false_positives(group_collection_predicted.get_group(group_name, id_predicted))

    return statistics



def print_mean_statistics(stats_list):
    d = {}
    overall_f1 = 0.0
    
    for slot_name in stats_list[0].statistics_dict:
        d[slot_name] = 0.0
     
    for stat in stats_list:
        for slot_name in stat.statistics_dict:
            d[slot_name] += stat.statistics_dict[slot_name].f1()
            
        overall_f1 += stat.get_micro_stats().f1()
    
    for group_name in ontology.group_names:
        print(group_name + ' --------------------------')
        
        for key in d:
            if key in ontology.group_slots[group_name]:
                d[key] /= len(stats_list)
                print(key + ': ' + "{:.4f}".format(d[key]))
        
    overall_f1 /= len(stats_list)
    print('Overall: ' + "{:.4f}".format(overall_f1))
    
    
    
    
ids = import_duplicated_trials_ids('duplicated_trials_ids.txt')
filename_pairs_triples = create_filename_pairs(ids, 'n-triples')
filename_pairs_annodb = create_filename_pairs(ids, 'annodb')
filename_pairs_tokenization = create_filename_pairs(ids, 'csv')

filename_pairs_tokenization = sort_filename_pairs(filename_pairs_tokenization, False)
filename_pairs_annodb = sort_filename_pairs(filename_pairs_annodb, False)
filename_pairs_triples = sort_filename_pairs(filename_pairs_triples, False)
stat = slot_filling_agreement(filename_pairs_tokenization, filename_pairs_annodb, filename_pairs_triples)


print_mean_statistics([stat])