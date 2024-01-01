import gc
from random import randrange
import sys
sys.setrecursionlimit(10**6)

from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

'''for item in dataset:    
    text = item['text']
    output = classifier(text, candidate_labels, multi_label=True)
    
    labels = output['labels']
    scores = output['scores']
    
    final_labels = []
    for x, y in zip(labels, scores):
        if (y > 0.5):
            final_labels.append(x)
    if (len(final_labels) == 0): continue
    item['feelings'] = final_labels

    write_single_dict_to_jsonl_file('dataset/final_dataset.json', item, file_access = 'a')'''

def annotate_single(item, output_file = 'dataset/final_dataset2.json'):
    print(item['id'])
    text = item['text']
    output = classifier(text, candidate_labels, multi_label=True)
    
    labels = output['labels']
    scores = output['scores']
    
    final_labels = []
    for x, y in zip(labels, scores):
        if (y > 0.5):
            final_labels.append(x)
    #if (len(final_labels) == 0): return
    item['emotions'] = final_labels

    
    write_single_dict_to_jsonl_file(output_file, item, file_access = 'a')
    gc.collect()

def collect_multi(max_workers = 1, input_file = 'dataset/eval_set.json', output_file = 'dataset/eval_set2.json'):

    dataset = read_list_from_jsonl_file(input_file)
    with ThreadPoolExecutor(max_workers = max_workers) as executor:
        results = executor.map(annotate_single, dataset, [output_file]*len(dataset), timeout = 600)

'''def filter_repetition(input_file = 'dataset/dataset_text.json'):

    dataset = read_list_from_jsonl_file(input_file)
    
    ids = []
    new_dataset = []
    for item in dataset:
        if (item['id'] in ids): continue
        new_dataset.append(item)
        ids.append(item['id'])
    
    write_list_to_jsonl_file('dataset/dataset_text_new.json', new_dataset)'''
    

#........................................
candidate_labels = ['anger', 'brain dysfunction (forget)', 'emptiness', 'hopelessness', 'lack of energy', 'loneliness', 'sadness', 'self hate', 'suicide intent', 'worthlessness']

model1 = 'facebook/bart-large-mnli' 
model2 = 'MoritzLaurer/mDeBERTa-v3-base-mnli-xnli' 
model3 = 'MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli'
model4 = 'cross-encoder/nli-deberta-v3-small'
model5 = 'SamLowe/roberta-base-go_emotions'

model_list = [model1, model2, model3, model4, model5]

global classifier

'''for idx, model in enumerate(model_list):
    classifier = pipeline("zero-shot-classification", model=model)
    collect_multi(input_file = 'dataset/dataset_text.json', output_file = 'dataset/final_dataset_' + str(idx + 1) + '.json')'''
    
classifier = pipeline("zero-shot-classification", model=model5)
collect_multi(input_file = 'dataset/dataset_text.json', output_file = 'dataset/final_dataset_' + str(5) + '.json')

'''if __name__ == "__main__":
    #collect_multi(input_file = 'dataset/eval_set.json', output_file = 'dataset/eval_set4.json')
    collect_multi(input_file = 'dataset/dataset_text.json', output_file = 'dataset/final_dataset.json')
    #filter_repetition()'''

