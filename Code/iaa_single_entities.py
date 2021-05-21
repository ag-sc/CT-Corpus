import sys,os,copy,pickle
import numpy as np
import random
MIN_LABEL_COUNTING = 20

NO_ANNOTATION_LABEL = 'O'

'''
all_classes = []
groups = set()

f = open('classes.csv')
for line in f:
	if line[0] == '#':
		print(line)
		continue
			
	cols = line.split('\t')
	all_classes.append(cols[0].strip())
		
		
f.close()
f = open('properties.csv')
for line in f:
	if line[0] == '#':
		print(line)
		continue			
		
	cols = line.split('\t')
	groups.add(cols[0].strip())

		
f = open('labels.txt', 'w')
for c in all_classes:
	if c not in groups:
		f.write(c + '\n')

f.close()
sys.exit()
'''
		


def import_categories(filename):
	categories = []
	
	with open(filename) as f:
		for line in f:
			categories.append(line.strip())
			
	return categories
	
	
	
def import_superclasses(filename):
	superclasses = {}
	
	with open(filename) as f:
		for line in f:
			if line[0] == '#':
				continue

			cols = line.split('\t')
			superclass = cols[0].strip()
			subclass = cols[1].strip()
			superclasses[subclass] = superclass
			
	return superclasses
	
	
	
def create_most_general_superclass_mapping(superclasses):
	mg_superclasses = copy.deepcopy(superclasses)
	subclasses = list(superclasses.keys())
	
	while True:
		modified = False
		
		for subclass in superclasses:
			superclass = mg_superclasses[subclass]
			
			if superclass in subclasses:
				superclass = superclasses[superclass]
				mg_superclasses[subclass] = superclass
				modified = True
				
		if not modified:
			break
		
	return mg_superclasses
	
	
	
def save_superclasses(superclasses, filename):
	f = open(filename, 'w')
	
	for subclass in superclasses:
		f.write(subclass + '\t' + superclasses[subclass] + '\n')
		
	f.close()
	
	

superclasses = import_superclasses('subclasses_new.csv')	
superclasses = create_most_general_superclass_mapping(superclasses)
save_superclasses(superclasses, 'most_general_superclasses.csv')
f = open('most_general_superclasses.dump', 'wb')
pickle.dump(superclasses, f)
f.close()

'''
f = open('most_general_superclasses.dump')
superclasses = pickle.load(f)
f.close()
'''	

def replace_labels_by_superclass(dataset, superclasses):
	result_dataset = {}
	
	for doc_id in dataset:
		doc = dataset[doc_id]
		result_doc = []

		for seq in doc:
			result_seq = []
			
			for token,labels in seq:
				result_labels = []
				
				for label in labels:
					if label in superclasses:
						result_labels.append(superclasses[label])
					else:
						result_labels.append(label)
						
				result_seq.append((token, result_labels))
				
			result_doc.append(result_seq)
			
		result_dataset[doc_id] = result_doc
		
	return result_dataset
	
	
	
def import_token_ranges(filename):
	sentences = []
	token_ranges = []
	current_sentence_number = 1
	
	with open(filename) as f:
		for line in f:
			if line[0] == '#':
				continue
				
			cols = line.split(',')
			sentence_number = int( cols[1].strip() )
			onset = int( cols[6].strip() )		
			offset = int( cols[7].strip() )
			token = cols[8].strip().lower()
			
			# new sentence ?
			if sentence_number != current_sentence_number:
				sentences.append(token_ranges)
				token_ranges = []
				current_sentence_number = sentence_number
				
			token_ranges.append( (onset,offset,token) )
		
	# last sentence
	sentences.append(token_ranges)
		
	return sentences	
	
	
	
def import_annotation_ranges(filename):
	annotation_ranges = []
	
	with open(filename) as f:
		for line in f:
			if line[0] == '#':
				continue
		
			cols = line.split(',')
			label =  cols[1].strip() 	
			
			try:
				onset = int( cols[2].strip() )		
				offset = int( cols[3].strip() )
			except ValueError:
				print('ValueError in import_annotation_ranges')
				continue
				
			annotation_ranges.append( (onset,offset,label) )
		
	return annotation_ranges	
	
	
	
def create_annotated_document(sentences, annotation_ranges, superclasses, iob_schema=False):
	annotated_sentences = []
	
	# create annotation lists
	for sentence in sentences:
		annotated_sentence = []
		
		for onset,offset,token in sentence:
			annotated_sentence.append( (token,[]) )
		
		annotated_sentences.append(annotated_sentence)
		
	# get annotations
	for annotation_onset,annotation_offset,original_label in annotation_ranges:
		iob_phase = 0
		if original_label in superclasses:
			original_label = superclasses[original_label]
		
		for sentence_index,sentence in enumerate(sentences):
			for token_index,(token_onset,token_offset,token) in enumerate(sentence):
				if token_onset >= annotation_onset and token_offset <= annotation_offset:
					if iob_schema:
						if iob_phase == 0:
							label = original_label + '-B'
							iob_phase = 1
						elif iob_phase == 1:
							label = original_label + '-I'
							iob_phase = 2
					else:
						label = original_label
							
					annotated_token,labels_list = annotated_sentences[sentence_index][token_index]
					labels_list.append(label)
					
					# check if tokens match
					if token != annotated_token:
						raise 'Tokens do not match! Token: ' + token + '; annotated token: ' + annotated_token
					
	# add O labels for tokens having no annotation
	for annotated_sentence in annotated_sentences:
		for annotated_token,labels_list in annotated_sentence:
			if len(labels_list) == 0:
				labels_list.append('O')

	return annotated_sentences



def shrink_annotation(annotated_sentences, category):
	result_sentences = []
	
	for annotated_sentence in annotated_sentences:
		result_sentence = []
		
		for token,labels in annotated_sentence:
			if category in labels:
				result_sentence.append( (token, category) )
			else:
				result_sentence.append( (token, 'O') )
				
		result_sentences.append(result_sentence)
		
	return result_sentences
	
	
	
def keep_labels(documents, all_labels):
	result_documents = {}
	
	for doc_id in documents:
		result_sentences = []

		for annotated_sentence in documents[doc_id]:
			result_sentence = []

			for token,labels in annotated_sentence:
				result_labels = []
				
				for label in labels:
					label_without_suffix = label.replace('-B', '').replace('-I', '')
					if label_without_suffix in all_labels:
						result_labels.append(label)
					elif label == 'O':
						result_labels.append('O')
						break
						
				result_sentence.append( (token,result_labels) )
			
			result_sentences.append(result_sentence)
		
		result_documents[doc_id] = result_sentences
		
	return result_documents
		
	
	
def create_label_seq(annotated_sentences):
	label_seq = []
	
	for annotated_sentence in annotated_sentences:
		for token,label in annotated_sentence:
			label_seq.append(label)
			
	return label_seq
	
	
	
def save_annotation_seq(annotated_sentences, filename):
	f = open(filename, 'w')
	
	for annotated_sentence in annotated_sentences:
		for token,labels in annotated_sentence:
			f.write(token + '\t' + str(labels) + '\n')
			
		#f.write('\n')
		
	f.close()
	
	
	
def extract_disease_ids(filenames, disease):
	ids = set()
	
	for filename in filenames:
		if disease in filename:
			cols = filename.split(' ')
			cols = cols[1].split('_')
			ids.add(cols[0])
			
	return ids
	
	
	
def import_annotated_documents(directory, disease, annotator_name, iob_schema=False):
	documents = {}
	filenames = os.listdir(directory)
	ids = extract_disease_ids(filenames, disease)
	
	for doc_id in ids:
		filename = disease + ' ' + doc_id + '_' + 'export.csv'
		token_ranges = import_token_ranges(directory + '/' + filename)
		
		filename = disease + ' ' + doc_id + '_' + annotator_name + '.annodb'
		annotation_ranges = import_annotation_ranges(directory + '/' + filename)
		
		annotated_document = create_annotated_document(token_ranges, annotation_ranges, superclasses, iob_schema)
		documents[doc_id] = annotated_document
		
	return documents
	
	
	
def observed_agreement(documents_annotator1, documents_annotator2, category):
	agreement_counter = 0.0
	n_tokens = 0
	
	for doc_id in documents_annotator1:
		annotated_sentences1 = documents_annotator1[doc_id]
		annotated_sentences2 = documents_annotator2[doc_id]
		
		annotated_sentences1 = shrink_annotation(annotated_sentences1, category)
		annotated_sentences2 = shrink_annotation(annotated_sentences2, category)
		
		labels1 = create_label_seq(annotated_sentences1)
		labels2 = create_label_seq(annotated_sentences2)
		
		if len(labels1) != len(labels2):
			raise 'ERROR: Sequences of unequal length'
		
		n_tokens += len(labels1)	
		for i in range(len(labels1)):
			if labels1[i] == labels2[i]:
				agreement_counter += 1
			
	return agreement_counter / n_tokens



def extract_labels_set(annotated_documents):
	labels_set = set()
	
	for doc_id in annotated_documents:
		annotated_sentences = annotated_documents[doc_id]
		for sentence in annotated_sentences:
			for token,labels in sentence:
				for label in labels:
					labels_set.add(label)
					
	return labels_set
		
		
		
def expected_agreement(documents_annotator1, documents_annotator2, category):	
	categories_dict1 = { category:0, 'O':0 }
	categories_dict2 = { category:0, 'O':0 }
	n_tokens = 0
	
	for doc_id in documents_annotator1:
		annotated_sentences1 = documents_annotator1[doc_id]
		annotated_sentences2 = documents_annotator2[doc_id]
		
		annotated_sentences1 = shrink_annotation(annotated_sentences1, category)
		annotated_sentences2 = shrink_annotation(annotated_sentences2, category)
		
		labels1 = create_label_seq(annotated_sentences1)
		labels2 = create_label_seq(annotated_sentences2)
		
		if len(labels1) != len(labels2):
			raise 'ERROR: Sequences of unequal length'
			
		n_tokens += len(labels1)
		
		for i in range(len(labels1)):	
			categories_dict1[labels1[i]] += 1
			categories_dict2[labels2[i]] += 1
		
	e = 0.0
	for cat in categories_dict1:
		e += categories_dict1[cat] * categories_dict2[cat]
		
	return e / n_tokens**2
	
	

def create_data_split_diabetes(documents, test_fraction):
	documents = [documents[doc_id] for doc_id in documents]
	num_train_docs = int( (1.0-test_fraction) * len(documents) )
	train_docs = documents[:num_train_docs]
	test_docs = documents[num_train_docs:]
	
	train_sentences = []
	for train_doc in train_docs:
		train_sentences.extend(train_doc)
		
	test_sentences = []
	for test_doc in test_docs:
		test_sentences.extend(test_doc)
	print(train_sentences[0])
	return train_sentences,test_sentences
	
	
	
def count_labels(directory, disease, annotator_name, superclasses):
	labels_counter = {}
	
	filenames = os.listdir(directory)
	ids = extract_disease_ids(filenames, disease)
	
	for doc_id in ids:
		filename = disease + ' ' + doc_id + '_' + annotator_name + '.annodb'
		annotation_ranges = import_annotation_ranges(directory + '/' + filename)
		
		for onset,offset,label in annotation_ranges:
			if label in superclasses:
				label = superclasses[label]
			if label in labels_counter:
				labels_counter[label] += 1
			else:
				labels_counter[label] = 1
				
	return labels_counter
	
	
	
def get_labels_above_threshold(labels_countings, threshold):
	labels = set()
	
	for label in labels_countings:
		if labels_countings[label] >= threshold:
			labels.add(label)
			
	return labels
	
	
	
def compute_f1(predicted_sequences, ground_truth_sequences):
	tps = 0.0
	fps = 0.0
	fns = 0.0
	
	
	
def compute_exact_match(predicted_sequences, ground_truth_sequences):
	num_total_units = 0.0
	num_matched_units = 0.0
	
	for seq_i,ground_truth_seq in enumerate(ground_truth_sequences):
		start_len_dict = {}
		unit_started = False
		current_unit_start_pos = None
		current_unit_length = None
		
		# get ground truth units for current sequence
		for i,label in enumerate(ground_truth_seq):
			if '-B' in label: # new unit starts
				if unit_started:
					start_len_dict[current_unit_start_pos] = current_unit_length
					num_total_units += 1

				current_unit_start_pos = i
				current_unit_length = 1
				unit_started = True
			elif '-I' in label:
				current_unit_length += 1
			elif label == 'O':
				if unit_started:
					start_len_dict[current_unit_start_pos] = current_unit_length
					num_total_units += 1
					unit_started = False
					
		if label != 'O':
			start_len_dict[current_unit_start_pos] = current_unit_length
			num_total_units += 1
					
		# check if predicted units match ground truth units
		predicted_seq = predicted_sequences[seq_i]
		for pos in start_len_dict:
			length = start_len_dict[pos]
			print('Current Pos: ' + str(pos))
			
			matched = True
			while length > 0:
				if predicted_seq[pos] != ground_truth_seq[pos]:
					matched = False
					break
				else:
					length -= 1
					pos += 1
					
			if matched:
				num_matched_units += 1
				print('match')
			else:
				print('no match')
				
	return num_matched_units / num_total_units
	
	
				
				
				
				
	
#documents_annotator1 = import_annotated_documents('data2', 'dm2', 'akramersunderbrink')	
#documents_annotator2 = import_annotated_documents('data2', 'dm2', 'kwoodley')
documents_annotator1 = import_annotated_documents('data3', 'gl', 'jshahinitiran')	
documents_annotator2 = import_annotated_documents('data3', 'gl', 'tstrakeljahn')




labels_count_glaucoma_jshahinitiran = count_labels('data3', 'gl', 'jshahinitiran', superclasses)	
labels_count_glaucoma_tstrakeljahn = count_labels('data3', 'gl', 'tstrakeljahn', superclasses)	

labels_count_diabetes_akramersunderbrink = count_labels('data3', 'dm2', 'akramersunderbrink', superclasses)
labels_count_diabetes_kwoodley = count_labels('data3', 'dm2', 'kwoodley', superclasses)

labels_glaucoma_original = set(labels_count_glaucoma_jshahinitiran.keys()).union(set(labels_count_glaucoma_tstrakeljahn.keys()))
labels_diabetes_original = set(labels_count_diabetes_akramersunderbrink.keys()).union(set(labels_count_diabetes_kwoodley.keys()))


#labels_diabetes = get_labels_above_threshold(labels_count_diabetes, 20)
#labels_glaucoma = get_labels_above_threshold(labels_count_glaucoma, 20)

#removed_labels_glaucoma = labels_glaucoma_original.difference(labels_glaucoma)
#removed_labels_diabetes = labels_diabtes_original.difference(labels_diabetes)

#labels_intersection = labels_diabetes.intersection(labels_glaucoma)


slots = []
domain_classes = ['Arm', 'ClinicalTrial', 'DiffBetweenGroups', 'Endpoint', 'EvidenceQuality', 'Intervention', 'Medication', 'Outcome', 'Population', 'Publication']

with open('properties.csv') as f:
	for line in f:
		line = line.strip()
		cols = line.split('\t')
		if cols[5] == 'TRUE':
			slots.append(cols[2])
		elif cols[2] not in domain_classes:
			slots.append(cols[2])
'''			
with open('slots.txt', 'w') as f:
	for slot in slots:
		if slot in superclasses:
			superclass = superclasses[slot]
		else:
			if slot in labels_intersection:
				superclass = slot
			else:
				superclass = 'None'
			
		f.write(slot + ': ' + superclass + '\n')
'''





'''
documents = import_annotated_documents('data', 'dm2', 'admin', True)	
documents = keep_labels(documents, labels_intersection)
train_sentences,test_sentences = create_data_split_diabetes(documents, 0.2)
random.shuffle(train_sentences)
random.shuffle(test_sentences)

with open('train_diabetes.dump', 'wb') as f:
	pickle.dump(train_sentences, f)
with open('test_diabetes.dump', 'wb') as f:
	pickle.dump(test_sentences, f)
	
sys.exit()
'''

'''
labels_count = sorted(labels_count.items(), key=lambda x: x[1]) 
for key,value in labels_count:
	print(key + ': ' + str(value))
	
sys.exit()





labels1 = extract_labels_set(documents_annotator1)
labels2 = extract_labels_set(documents_annotator2)
labels_dm = labels1.union(labels2)

documents_annotator1 = import_annotated_documents('data', 'gl', 'jshahinitiran')	
documents_annotator2 = import_annotated_documents('data', 'gl', 'tstrakeljahn')
labels1 = extract_labels_set(documents_annotator1)
labels2 = extract_labels_set(documents_annotator2)
labels_gl = labels1.union(labels2)

labels = labels_dm.union(labels_gl)

superclasses = import_superclasses('subclasses_new.csv')	
superclasses = create_most_general_superclass_mapping(labels, superclasses)
c = set(superclasses.values())
print(c)
print(len(c))
f = open('most_general_superclasses.dump', 'wb')
pickle.dump(superclasses, f)
f.close()
sys.exit()
'''
documents_annotator1 = replace_labels_by_superclass(documents_annotator1, superclasses)
documents_annotator2 = replace_labels_by_superclass(documents_annotator2, superclasses)
categories = extract_labels_set(documents_annotator1)
categories = categories.union(extract_labels_set(documents_annotator2))

categories_to_remove = set(['ConfInterval', 'SdError', 'NumberPatients', 'PercentagePatients', 'PValue', 'SdDeviation', 
                            'Age', 'ApplicationCondition', 'DiffBetweenGroups', 'Interval', 'Ocular_Allergy'])
labels_iaa = labels_glaucoma_original - categories_to_remove

print(len(categories))
iaa_values = {}
for category in labels_iaa:
	if category == 'O':
		continue
		
	o = observed_agreement(documents_annotator1, documents_annotator2, category)
	e = expected_agreement(documents_annotator1, documents_annotator2, category)
	
	try:
		agreement = (o-e) / (1-e)
	except ZeroDivisionError:
		agreement = (o-e) / (0.00001)
    
	iaa_values[category] = agreement	
    
for k,v in sorted(iaa_values.items(), key=lambda (k,v): v):
    print(k + ': ' + "{:.4f}".format(v))
    
	
l = np.array(list(iaa_values.values()))
mean = np.mean(l)
print('mean: ' + "{:.4f}".format(mean))


	

