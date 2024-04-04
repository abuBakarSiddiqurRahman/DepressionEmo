import argparse
import datasets
import evaluate
import glob
import matplotlib.pyplot as plt
import nltk
import numpy as np
import os
import pandas as pd
import random
import re
import torch

from datasets import Dataset, load_dataset, concatenate_datasets
from file_io import *
from huggingface_hub import HfFolder
from nltk.tokenize import sent_tokenize
from sklearn.metrics import f1_score, recall_score, precision_score
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

nltk.download("punkt")
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
print('CUDA: ', torch.cuda.is_available())

emotion_list = ['anger', 'brain dysfunction (forget)', 'emptiness', 'hopelessness', 'loneliness', 'sadness', 'suicide intent', 'worthlessness']

def preprocess_function(sample, padding="max_length"):
    # add prefix to the input for t5
    inputs = [item for item in sample["text"]]

    # tokenize inputs
    model_inputs = tokenizer(inputs, max_length=max_source_length, padding=padding, truncation=True)
    
    # tokenize labels
    sample["label_id"] = [str(x) for x in sample["label_id"]]
    labels = tokenizer(text_target=sample["label_id"], max_length=max_target_length, padding=padding, truncation=True)
    
  
    # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
    # padding in the loss. 
    if padding == "max_length":
        labels["input_ids"] = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
        ]

    model_inputs["labels"] = labels["input_ids"]
    
    return model_inputs

# helper function to postprocess text
def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    # rougeLSum expects newline after each sentence
    preds = ["\n".join(sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(sent_tokenize(label)) for label in labels]

    return preds, labels

def compute_metrics(eval_preds):
    
    preds, labels = eval_preds
    #if isinstance(preds, tuple): preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True, clean_up_tokenization_spaces = False)
    
    # replace -100 in the labels as we can't decode them.
    #print('tokenizer.pad_token_id: ', tokenizer.pad_token_id)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
    
    #predictions = ["hello there", "general kenobi"]
    #references = ["hello there", "general kenobi"]
    #rouge = rouge.compute(predictions=decoded_preds, references=decoded_labels)
    
    digits = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    #print('decoded_labels: ', decoded_labels[0])
    #decoded_labels = [x.replace(' ', '') for x in decoded_labels] # remove spaces
    #print('decoded_labels: ', decoded_labels[0])
    
    decoded_labels = [[int(y) if y in digits else 0 for y in x] for x in decoded_labels]
    decoded_labels = [[0]*(len(emotion_list)-len(x)) + x for x in decoded_labels] # if generate not enough
    decoded_labels = [x[0:len(emotion_list)] for x in decoded_labels]
    decoded_labels = [[0 if y == 0 else 1 for y in x] for x in decoded_labels]
    #print('decoded_labels: ', decoded_labels[0])
    
    #print('decoded_preds: ', decoded_preds[0])
    #decoded_preds = [x.replace(' ', '') for x in decoded_preds] # remove spaces
    decoded_preds = [[int(y) if y in digits else 0 for y in x] for x in decoded_preds]
    
    decoded_preds = [[0]*(len(emotion_list)-len(x)) + x for x in decoded_preds] # if generate not enough
    decoded_preds = [x[0:len(emotion_list)] for x in decoded_preds]
    decoded_preds = [[0 if y == 0 else 1 for y in x] for x in decoded_preds]
    #print('decoded_preds: ', decoded_preds[0])
    
    f1_mi = f1_score(y_true=decoded_labels, y_pred=decoded_preds, average='micro')
    re_mi = recall_score(y_true=decoded_labels, y_pred=decoded_preds, average='micro')
    pre_mi = precision_score(y_true=decoded_labels, y_pred=decoded_preds, average='micro')
    
    f1_mac = f1_score(y_true=decoded_labels, y_pred=decoded_preds, average='macro')
    re_mac = recall_score(y_true=decoded_labels, y_pred=decoded_preds, average='macro')
    pre_mac = precision_score(y_true=decoded_labels, y_pred=decoded_preds, average='macro')
    
    result = {}
    result['f1_micro'] = f1_mi
    result['recall_micro'] = re_mi
    result['precision_micro'] = pre_mi
    
    result['f1_macro'] = f1_mac
    result['recall_macro'] = re_mac
    result['precision_macro'] = pre_mac
   
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    
    return result


def train(train_set, val_set, test_set, tokenizer, model, model_name = 'facebook/bart-base', max_source_length = 64, max_target_length = 8, epochs = 40, batch_size = 4):
    
    # load dataset 
    '''train_set = datasets.load_dataset('json', data_files = 'dataset/train.json', split="train")
    test_set = datasets.load_dataset('json', data_files = 'dataset/test.json', split="train")
    val_set = datasets.load_dataset('json', data_files = 'dataset/val.json', split="train")

    print(f"Train dataset size: {len(train_set)}")
    print(f"Test dataset size: {len(test_set)}")
    print(f"Val dataset size: {len(val_set)}")'''

    # Load tokenizer
    #tokenizer = AutoTokenizer.from_pretrained(model_name)

    #dataset['train'] = dataset['train'].shuffle(seed=42).select(range(2000)) 
    #dataset['test'] = dataset['test'].shuffle(seed=42).select(range(1000)) 
    #dataset['train'] = dataset['train'].shuffle(seed=42)

    train_df = pd.DataFrame(train_set)
    val_df = pd.DataFrame(val_set)
    test_df = pd.DataFrame(test_set)

    # The maximum total input sequence length after tokenization. 
    # Sequences longer than this will be truncated, sequences shorter will be padded.
    tokenized_inputs = concatenate_datasets([train_set, test_set]).map(lambda x: tokenizer(x["text"], truncation=True), batched=True, remove_columns=['id', 'title', 'post', 'upvotes', 'emotions', 'date', 'text', 'label_id'])
    print(f"Max source length: {max_source_length}")

    # The maximum total sequence length for target text after tokenization. 
    # Sequences longer than this will be truncated, sequences shorter will be padded."
    tokenized_targets = concatenate_datasets([train_set, test_set]).map(lambda x: tokenizer(str(x["label_id"]), truncation=True), batched=True, remove_columns=['id', 'title', 'post', 'upvotes', 'emotions', 'date', 'text', 'label_id'])

    #max_target_length = max([len(str(x)) for x in tokenized_targets["input_ids"]])

    print('Tokenized targets: ', tokenized_targets)
    print(f"Max target length: {max_target_length}")

    tokenized_train_dataset = train_set.map(preprocess_function, batched=True, remove_columns=['id', 'title', 'post', 'upvotes', 'emotions', 'date', 'text', 'label_id'])
    print(f"Keys of tokenized dataset: {list(tokenized_train_dataset.features)}")

    tokenized_val_dataset = val_set.map(preprocess_function, batched=True, remove_columns=['id', 'title', 'post', 'upvotes', 'emotions', 'date', 'text', 'label_id'])
    print(f"Keys of tokenized dataset: {list(tokenized_val_dataset.features)}")
    
    # load model from the hub
    #model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    #model = model.to(device)

    # Metric
    metric_f1 = evaluate.load("f1")
    metric_pre = evaluate.load("precision")
    metric_re = evaluate.load("recall")
    metric_rouge = evaluate.load("rouge")

    # we want to ignore tokenizer pad token in the loss
    label_pad_token_id = -100
    # data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8
        )

    # Hugging Face repository id
    repository_id = ''
    try:
        repository_id = f"{model_name.split('/')[1]}"
    except:
        repository_id = f"{model_name}"

    # define training args
    training_args = Seq2SeqTrainingArguments(
        gradient_accumulation_steps = 4,
        #gradient_checkpointing=True,
        output_dir=repository_id,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        predict_with_generate=True,
        fp16=True, # overflows with fp16, for T5 models please hide this parameter
        #learning_rate=3e-4,
        num_train_epochs=epochs,
        # logging & evaluation strategies
        logging_dir=f"{repository_id}/logs",
        logging_strategy="epoch", 
        # logging_steps=1000,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=False,
        metric_for_best_model="eval_f1_macro",
        # push to hub parameters
        report_to="tensorboard",
        generation_max_length = max_target_length,
        #push_to_hub=True,
        #hub_strategy="every_save",
        #hub_model_name=repository_id,
        #hub_token=HfFolder.get_token(),
        )

    # create Trainer instance
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_val_dataset,
        compute_metrics=compute_metrics,
        )

    trainer.train()
    #trainer.evaluate()

def test(dataset, model_name, model, tokenizer, input_file = 'dataset/test.json', \
                     batch_size = 4, max_len = 128, min_len = 1):

    
    if (len(dataset) == 0): # load dataset if not given
        dataset = read_list_from_jsonl_file(input_file)
        
    pred_list = []
    for i in range(0, len(dataset), batch_size):

        n_batch = 0
        if (len(dataset)%batch_size != 0): n_batch = len(dataset)//batch_size + 1
        else: n_batch = len(dataset)//batch_size

        sys.stdout.write('Infer batch: %d/%d \t Model: %s \r' % (i//batch_size + 1, n_batch, model_name))
        #sys.stdout.flush()
        
        subset = dataset[i:i + batch_size]
        texts = [item['text'] for item in subset]
        
        #print(texts)
        
        inputs = tokenizer(texts, padding = "max_length", truncation = True, max_length = max_len, \
                           return_tensors = 'pt').to(device)
                           
        outputs = []
        with torch.no_grad():
            outputs = model.generate(**inputs, max_length = max_len, min_length = min_len, \
                                         num_beams = 4, do_sample = False, return_dict_in_generate = True, output_scores = True)  

        preds = tokenizer.batch_decode(outputs.sequences, skip_special_tokens = True)
        
        preds = [[x] for x in preds]
        pred_list += preds

    pred_list = [[x for x in pred][0].strip() for pred in pred_list] # use strip() to remove spaces
    
    label_list = [str(item['label_id']) for item in dataset]
    
    digits = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    # ['hola' if i == 1 else '' for i in list_num]
    label_list = [[int(y) if y in digits else 0 for y in x] for x in label_list]
    label_list = [[0]*(len(emotion_list)-len(x)) + x for x in label_list] # if generate not enough
    label_list = [x[0:len(emotion_list)] for x in label_list]
    label_list = [[0 if y == 0 else 1 for y in x] for x in label_list]
    print('label_list: ', label_list)
    
    pred_list = [[int(y) if y in digits else 0 for y in x] for x in pred_list]
    pred_list = [[0]*(len(emotion_list)-len(x)) + x for x in pred_list] # if generate not enough
    pred_list = [x[0:len(emotion_list)] for x in pred_list]
    pred_list = [[0 if y == 0 else 1 for y in x] for x in pred_list]
    print('pred_list: ', pred_list)
    
    f1_mi = f1_score(y_true=label_list, y_pred=pred_list, average='micro')
    re_mi = recall_score(y_true=label_list, y_pred=pred_list, average='micro')
    pre_mi = precision_score(y_true=label_list, y_pred=pred_list, average='micro')
    
    f1_mac = f1_score(y_true=label_list, y_pred=pred_list, average='macro')
    re_mac = recall_score(y_true=label_list, y_pred=pred_list, average='macro')
    pre_mac = precision_score(y_true=label_list, y_pred=pred_list, average='macro')
    
    result = {}
    result['f1_micro'] = f1_mi
    result['recall_micro'] = re_mi
    result['precision_micro'] = pre_mi
    
    result['f1_macro'] = f1_mac
    result['recall_macro'] = re_mac
    result['precision_macro'] = pre_mac
    
    print('Emotion: All')
    print(str(round(f1_mac, 2)) + ' & ' + str(round(pre_mac, 2)) + ' & ' + str(round(re_mac, 2)) + ' & ' + str(round(f1_mi, 2)) + ' & ' + str(round(pre_mi, 2)) + ' & ' + str(round(re_mi, 2)))
    print('----------------------------------------')
    
    for index, emotion in enumerate(emotion_list):
    
        print('emotion: ', emotion)
        
        temp_label_list = [x[index] for x in label_list]
        temp_pred_list = [x[index] for x in pred_list]
        
        f1_mi = f1_score(y_true=temp_label_list, y_pred=temp_pred_list, average='micro')
        re_mi = recall_score(y_true=temp_label_list, y_pred=temp_pred_list, average='micro')
        pre_mi = precision_score(y_true=temp_label_list, y_pred=temp_pred_list, average='micro')
    
        f1_mac = f1_score(y_true=temp_label_list, y_pred=temp_pred_list, average='macro')
        re_mac = recall_score(y_true=temp_label_list, y_pred=temp_pred_list, average='macro')
        pre_mac = precision_score(y_true=temp_label_list, y_pred=temp_pred_list, average='macro')
    
        result['f1_micro_' + emotion] = f1_mi
        result['recall_micro_' + emotion] = re_mi
        result['precision_micro_' + emotion] = pre_mi
    
        result['f1_macro_' + emotion] = f1_mac
        result['recall_macro_' + emotion] = re_mac
        result['precision_macro_' + emotion] = pre_mac
        
        print(str(round(f1_mac, 2)) + ' & ' + str(round(pre_mac, 2)) + ' & ' + str(round(re_mac, 2)) + ' & ' + str(round(f1_mi, 2)) + ' & ' + str(round(pre_mi, 2)) + ' & ' + str(round(re_mi, 2)))
        print('----------------------------------------')
        
    print('result: ', result)
    return result

def main(args):
    if (args.mode == 'train'):
        train_set = datasets.load_dataset('json', data_files = args.train_path, split="train")
        test_set = datasets.load_dataset('json', data_files = args.test_path, split="train")
        val_set = datasets.load_dataset('json', data_files = args.val_path, split="train")

        print(f"Train dataset size: {len(train_set)}")
        print(f"Test dataset size: {len(test_set)}")
        print(f"Val dataset size: {len(val_set)}")

        model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)
        model.to(device)
        
        train(train_set, val_set, test_set, tokenizer, model, model_name = args.model_name, max_source_length = args.max_source_length, max_target_length = args.max_target_length, epochs = args.epochs, batch_size = args.batch_size)
    
    elif (args.mode == 'test'):
        model = AutoModelForSeq2SeqLM.from_pretrained(args.model_path)
        model.to(device)
        model.eval()
        test([], args.model_name, model, tokenizer, input_file = args.test_path, batch_size = args.test_batch_size, max_len = args.max_source_length, min_len = args.min_target_length)

#...............................................................................            
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Training Parameter')
    parser.add_argument('--mode', type=str, default='train') # or test
    parser.add_argument('--model_name', type=str, default='facebook/bart-base') # or test
    parser.add_argument('--train_path', type=str, default='dataset/train.json') 
    parser.add_argument('--test_path', type=str, default='dataset/test.json')
    parser.add_argument('--val_path', type=str, default='dataset/val.json')
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--test_batch_size', type=int, default=16)
    parser.add_argument('--max_source_length', type=int, default=256)
    parser.add_argument('--max_target_length', type=int, default=8)
    parser.add_argument('--min_target_length', type=int, default=1)
    
    parser.add_argument('--model_path', type=str, default='bart-base\checkpoint-452')
    parser.add_argument('--test_file', type=str, default='dataset/test.json')
  
    args = parser.parse_args()

    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, do_lower_case = False, add_prefix_space = True)
    
    global max_source_length
    max_source_length = args.max_source_length
    
    global max_target_length
    max_target_length = args.max_target_length
    
    main(args)
    
# python seq2seq.py --mode "train" --model_name "facebook/bart-base" --train_path "dataset/train.json" --val_path "dataset/val.json" --test_path "dataset/test.json" --epochs 25 --batch_size 4 --max_source_length 256
# python seq2seq.py --mode "test" --model_name "facebook/bart-base" --model_path "bart-base\checkpoint-1321" --test_path "dataset/test.json" --test_batch_size 4 --max_source_length 256 --min_target_length 1
        
