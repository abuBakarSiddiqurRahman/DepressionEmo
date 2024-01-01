import argparse
import gc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import seaborn as sns
import sys
import torch
import transformers

from collections import defaultdict
from file_io import *
from matplotlib import rc
from pylab import rcParams
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import train_test_split
from textwrap import wrap
from torch import nn, optim, functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
emotion_list = ['anger', 'brain dysfunction (forget)', 'emptiness', 'hopelessness', 'loneliness', 'sadness', 'suicide intent', 'worthlessness']

class PreparedDataset():
    def __init__(self, texts, categories, tokenizer, max_len):
        self.texts = texts
        self.categories = categories
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = str(self.texts[item])
        category = self.categories[item]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            #pad_to_max_length=True,
            padding = "max_length",
            return_attention_mask=True,
            truncation=True,
            return_tensors='pt',
            )

        return {
                'text': text,
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
                'categories': torch.tensor(category, dtype=torch.long)
            }

class CategoryClassifier(nn.Module):
    def __init__(self, n_classes, pretrained_model = 'bert-base-cased'):
        super(CategoryClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(pretrained_model)
        self.drop = nn.Dropout(p=0.1)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask, return_dict=False):
        _, pooled_output = self.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=return_dict)
        output = self.drop(pooled_output)
        return self.out(output)

def search_index(type_list, class_names):

    for index, c in enumerate(class_names):
        c_list = c
        
        if (type(c) is tuple): c_list = [*c]
        else: c_list = [c_list]
        
        #print('-- c_list: ', c_list)
        n_intersection = len(set(c_list).intersection(set(type_list)))
        #print('-- n_intersection: ', n_intersection)
        if (n_intersection == len(c_list) and len(c_list) == len(type_list)): return index

    return -1

def create_data_loader(dataset, tokenizer, class_names, max_len, batch_size):

    texts, categories = [], []
 
    for item in dataset:
        item_classifier = [str(item['label_id'])]
        index = search_index(item_classifier, class_names)
        if (index == -1): continue
        categories.append(index)
        texts.append(item['text'])
    
    #print('categories: ', categories)
    ds = PreparedDataset(texts=np.array(texts),
                         categories=np.array(categories),
                         tokenizer=tokenizer,
                         max_len=max_len)
    return DataLoader(ds, batch_size=batch_size, num_workers=4)


def train_epoch(model, data_loader, loss_fn, optimizer, device, scheduler, n_examples, class_names):

    model = model.train()
    losses = []
    correct_predictions = 0
    i = 0
    pred_list = []
    true_list = []
    
    for d in data_loader:
        
        sys.stdout.write('Training batch: %d/%d \r' % (i, len(data_loader)))
        #sys.stdout.flush()
        
        i = i + 1
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        categories = d["categories"].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        _, preds = torch.max(outputs, dim=1)
        loss = loss_fn(outputs, categories)
        
        correct_predictions += torch.sum(preds == categories)
        losses.append(loss.item())
        pred_list.append(preds.tolist())
        true_list.append(categories.tolist())
            
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        
    pred_list = sum(pred_list, [])
    true_list = sum(true_list, [])
    
    pred_list = [class_names[c] for c in pred_list]
    true_list = [class_names[c] for c in true_list]
    
    pred_list = convert_labels(pred_list)
    true_list = convert_labels(true_list)
    
    #print('pred_list: ', pred_list)
    #print('true_list: ', true_list)
    
    f1_mi = f1_score(y_true=true_list, y_pred=pred_list, average='micro')
    re_mi = recall_score(y_true=true_list, y_pred=pred_list, average='micro')
    pre_mi = precision_score(y_true=true_list, y_pred=pred_list, average='micro')
    
    f1_mac = f1_score(y_true=true_list, y_pred=pred_list, average='macro')
    re_mac = recall_score(y_true=true_list, y_pred=pred_list, average='macro')
    pre_mac = precision_score(y_true=true_list, y_pred=pred_list, average='macro')
    
    result = {}
    result['f1_micro'] = f1_mi
    result['recall_micro'] = re_mi
    result['precision_micro'] = pre_mi
    
    result['f1_macro'] = f1_mac
    result['recall_macro'] = re_mac
    result['precision_macro'] = pre_mac

    return result, np.mean(losses)
        
    #return correct_predictions.double() / n_examples, np.mean(losses)


def convert_labels(labels):
    
    labels2 = []
    for idx, label in enumerate(labels):
        
        temp = []
        num = len(emotion_list) - len(label)
        if (num == 0): 
            temp = [int(x) for x in label]
        else:
            temp = ''.join(['0']*num) + str(label)   
            temp = [int(x) for x in temp]
            
        labels2.append(temp)
        
    return labels2
    
def eval_model(model, data_loader, loss_fn, device, n_examples, class_names):
    model = model.eval()
    losses = []
    
    pred_list = []
    true_list = []
    
    with torch.no_grad():
        for d in data_loader:
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            categories = d["categories"].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            _, preds = torch.max(outputs, dim=1)
            loss = loss_fn(outputs, categories)
            pred_list.append(preds.tolist())
            true_list.append(categories.tolist())
            losses.append(loss.item())
   
    pred_list = sum(pred_list, [])
    true_list = sum(true_list, [])
    
    pred_list = [class_names[c] for c in pred_list]
    true_list = [class_names[c] for c in true_list]
    
    #print('pred_list: ', pred_list)
    #print('true_list: ', true_list)
    
    pred_list = convert_labels(pred_list)
    true_list = convert_labels(true_list)
    
    f1_mi = f1_score(y_true=true_list, y_pred=pred_list, average='micro')
    re_mi = recall_score(y_true=true_list, y_pred=pred_list, average='micro')
    pre_mi = precision_score(y_true=true_list, y_pred=pred_list, average='micro')
    
    f1_mac = f1_score(y_true=true_list, y_pred=pred_list, average='macro')
    re_mac = recall_score(y_true=true_list, y_pred=pred_list, average='macro')
    pre_mac = precision_score(y_true=true_list, y_pred=pred_list, average='macro')
    
    result = {}
    result['f1_micro'] = f1_mi
    result['recall_micro'] = re_mi
    result['precision_micro'] = pre_mi
    
    result['f1_macro'] = f1_mac
    result['recall_macro'] = re_mac
    result['precision_macro'] = pre_mac

    return result, np.mean(losses)

def get_predictions(model, data_loader):
    model = model.eval()
    sentences = []
    predictions = []
    prediction_probs = []
    real_values = []

    with torch.no_grad():
        for d in data_loader:
            texts = d["text"]
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            categories = d["categories"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            _, preds = torch.max(outputs, dim=1)
            sentences.extend(texts)
            predictions.extend(preds)
            prediction_probs.extend(outputs)
            real_values.extend(categories)

    predictions = torch.stack(predictions).cpu()
    prediction_probs = torch.stack(prediction_probs).cpu()
    real_values = torch.stack(real_values).cpu()
    return sentences, predictions, prediction_probs, real_values


def train_model(train_set, val_set, class_names = [], pretrained_model = 'bert-base-cased', 
                saved_model_file = 'best_model_state.bin', saved_history_file = 'history_file.json', epochs = 40, max_len = 256, batch_size = 8):

    # prepare dataset
    tokenizer = BertTokenizer.from_pretrained(pretrained_model)

    train_data_loader = create_data_loader(train_set, tokenizer, class_names, max_len, batch_size)
    val_data_loader = create_data_loader(val_set, tokenizer, class_names, max_len, batch_size)
    #test_data_loader = create_data_loader(df_test, tokenizer, class_names, max_len, batch_size)

    # create model
    model = CategoryClassifier(len(class_names), pretrained_model)
    model = model.to(device)

    data = next(iter(train_data_loader))
    input_ids = data['input_ids'].to(device)
    attention_mask = data['attention_mask'].to(device)
    #F.softmax(model(input_ids, attention_mask), dim=1)
    
    # lr=2e-5
    optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=True)
    total_steps = len(train_data_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
        )
    loss_fn = nn.CrossEntropyLoss().to(device)

    history = defaultdict(list)
    best_metric = 0
    best_epoch = -1
    write_single_dict_to_json_file(saved_history_file, {}, "w")
    
    for epoch in range(epochs):
        print('-' * 50)
        print(f'Epoch {epoch + 1}/{epochs}')
        

        train_result, train_loss = train_epoch(model, train_data_loader, loss_fn, optimizer, device, scheduler, len(train_set), class_names)
        train_f1_macro = train_result['f1_macro']
        print(f'Train loss: {train_loss}, Train f1 macro: {train_f1_macro}')
        
        val_result, val_loss = eval_model(model, val_data_loader, loss_fn, device, len(val_set), class_names)
        f1_macro = val_result['f1_macro']
        print(f'Val loss {val_loss}, Val f1 macro: {f1_macro}')
        
        history['train_result'].append(train_result)
        history['train_loss'].append(train_loss)
        history['val_result'].append(val_result)
        history['val_loss'].append(val_loss)
      
        if f1_macro > best_metric:
            torch.save(model.state_dict(), saved_model_file)
            best_metric = f1_macro
            best_epoch = epoch + 1

        # save history step
        print('val_f1_macro: ', f1_macro)
        print('train_f1_mac: ', train_f1_macro)

        history_dict = {}
        history_dict['epoch'] = str(epoch + 1)
        history_dict['best_epoch'] = str(best_epoch)
        history_dict['best_metric'] = str(best_metric)
        history_dict['train_result'] = train_result
        history_dict['train_loss'] = train_loss
        history_dict['val_result'] = val_result
        history_dict['val_loss'] = val_loss
        write_single_dict_to_json_file(saved_history_file, history_dict)
        
        torch.cuda.empty_cache()
        gc.collect()
        print('-' * 50)
    
 
def test_dataset(test_set, class_names = [],
                     pretrained_model = 'bert-base-cased', saved_model_file = 'best_bert_model.bin',
                     saved_history_file = 'best_bert_model.json', max_len = 256, batch_size = 8):
    
    tokenizer = BertTokenizer.from_pretrained(pretrained_model)

    #train_data_loader = create_data_loader(train_set, tokenizer, class_names, max_len, batch_size)
    #val_data_loader = create_data_loader(val_set, tokenizer, class_names, max_len, batch_size)
    test_data_loader = create_data_loader(test_set, tokenizer, class_names, max_len, batch_size)

    loss_fn = nn.CrossEntropyLoss().to(device)

    model = CategoryClassifier(len(class_names), pretrained_model)
    model.load_state_dict(torch.load(saved_model_file))
    model = model.to(device)
    
    texts, pred_list, pred_probs, true_list = get_predictions(model, test_data_loader)
   
    pred_list = pred_list.tolist()
    true_list = true_list.tolist()
    pred_list = [class_names[c] for c in pred_list]
    true_list = [class_names[c] for c in true_list]
    
    pred_list = convert_labels(pred_list)
    true_list = convert_labels(true_list)
    
    f1_mi = f1_score(y_true=true_list, y_pred=pred_list, average='micro')
    re_mi = recall_score(y_true=true_list, y_pred=pred_list, average='micro')
    pre_mi = precision_score(y_true=true_list, y_pred=pred_list, average='micro')
    
    f1_mac = f1_score(y_true=true_list, y_pred=pred_list, average='macro')
    re_mac = recall_score(y_true=true_list, y_pred=pred_list, average='macro')
    pre_mac = precision_score(y_true=true_list, y_pred=pred_list, average='macro')
    
    result = {}
    result['f1_micro'] = f1_mi
    result['recall_micro'] = re_mi
    result['precision_micro'] = pre_mi
    
    result['f1_macro'] = f1_mac
    result['recall_macro'] = re_mac
    result['precision_macro'] = pre_mac
    
    print('result: ', result)
    return result

def predict_single(text, tokenizer, model, max_len = 256):

    encoded_question = tokenizer.encode_plus(text,
                                           max_length=max_len,
                                           add_special_tokens=True,
                                           return_token_type_ids=False,
                                           #pad_to_max_length=True,
                                           padding = "max_length",
                                           return_attention_mask=True,
                                           return_tensors='pt',
                                           )

    input_ids = encoded_question['input_ids'].to(device)
    attention_mask = encoded_question['attention_mask'].to(device)
    output = model(input_ids, attention_mask)
    _, prediction = torch.max(output, dim=1)
    #print(f'Text: {text}')
    #print(f'Category: {class_names[prediction]}')
    return prediction


def predict_dataset(dataset, class_names, classifier_type = 'category',
                    pretrained_model = 'bert-base-cased',
                    model_file_name = 'best_bert_model_state_category_string.bin',
                    out_file_name = 'wikidata//task1_wikidata_test_pred.json'):

    tokenizer = BertTokenizer.from_pretrained(pretrained_model)
    model = CategoryClassifier(len(class_names), pretrained_model)
    model.load_state_dict(torch.load(model_file_name))
    model = model.to(device)

    data_list = []
    for item in dataset:

        if (item['question'] is None or item['question'].strip() == ''):
            item['category'] = ''
            item['type'] = []
        else:

            index_pred = predict_single_question(item['question'], tokenizer, model)
            name_pred = class_names[index_pred]

            if (classifier_type == 'category_string'):
                if (name_pred == 'boolean'):
                    item['category'] = 'boolean'
                    item['type'] = ['boolean']
                elif('literal,' in name_pred):
                    item['category'] = 'literal'
                    item['type'] = [name_pred.split(',')[1]]
                else:
                    item['category'] = name_pred
                
            else:
                if (item['category'] != 'boolean' and item['category'] != 'literal'):
                    name_pred_list = name_pred.split(",")
                    name_pred_list = [n.strip() for n in name_pred_list if n.strip() != '']
                    item['type'] = name_pred_list

        data_list.append(item)
    
    # write dict
    write_list_to_json_file(out_file_name, data_list, 'w')


def classifier_by_text(dataset):

    type_dict = {}
    for item in dataset:
        temp_type = str(item['label_id'])
        if (temp_type not in type_dict): type_dict[temp_type] = 1
        else: type_dict[temp_type] +=1
   
    type_dict = sorted(type_dict.items(), key = lambda x: x[1], reverse = True)
    return type_dict

def main(args):
    if (args.mode == 'train'):
        train_set = read_list_from_jsonl_file(args.train_path)
        val_set = read_list_from_jsonl_file(args.val_path)
        test_set = read_list_from_jsonl_file(args.test_path)
    
        dataset = train_set + val_set + test_set # capture all labels

        class_names = sorted(list(set([item[0] for item in classifier_by_text(dataset)])), key = lambda x: x)
        class_names = [c.strip() for c in class_names if c.strip() != '']
        #print('class_names: ', class_names)
    
        train_model(train_set, val_set, class_names, pretrained_model = args.model_name,
                     saved_model_file = 'best_bert_model.bin',
                     saved_history_file = 'best_bert_model.json', epochs = args.epochs)
    
    elif (args.mode == 'test'):
        
        train_set = read_list_from_jsonl_file(args.train_path)
        val_set = read_list_from_jsonl_file(args.val_path)
        test_set = read_list_from_jsonl_file(args.test_path)
    
        dataset = train_set + val_set + test_set # capture all labels

        class_names = sorted(list(set([item[0] for item in classifier_by_text(dataset)])), key = lambda x: x)
        class_names = [c.strip() for c in class_names if c.strip() != '']
        
        test_dataset(test_set, class_names = class_names,
                     pretrained_model = args.model_name, saved_model_file = 'best_bert_model.bin',
                     saved_history_file = 'best_bert_model.json', max_len = args.max_length, batch_size = args.test_batch_size)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Training Parameters')
    parser.add_argument('--mode', type=str, default='train') # or test
    parser.add_argument('--model_name', type=str, default='bert-base-cased') # or test
    parser.add_argument('--train_path', type=str, default='dataset/train.json') 
    parser.add_argument('--test_path', type=str, default='dataset/test.json')
    parser.add_argument('--val_path', type=str, default='dataset/val.json')
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--test_batch_size', type=int, default=16)
    parser.add_argument('--max_length', type=int, default=256)
    parser.add_argument('--model_path', type=str, default='bart-base\checkpoint-452')
    parser.add_argument('--test_file', type=str, default='dataset/test.json')
  
    args = parser.parse_args()
   
    main(args)
    
# python bert.py  --mode "train" --model_name "bert-base-cased" --epochs 25 --batch_size 8 --max_length 256 --train_path "dataset/train.json" --val_path "dataset/val.json" --test_path "dataset/test.json"

# python bert.py --mode "test" --train_path "dataset/train.json" --val_path "dataset/val.json" --test_path "dataset/test.json" --max_length 300 --test_batch_size 16