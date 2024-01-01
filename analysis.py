from file_io import *
from datetime import datetime

import numpy as np  
import matplotlib.pyplot as plt  

import spacy
nlp = spacy.load('en_core_web_md')
import pandas as pd

stop_words = read_list_from_text_file('dataset/stopwords-en.txt')
stop_words.extend(["n’t", "’s", "'s", "life", "people", "'m", "’m", "n't", "“", "”", "nt", "’ve", "...", "happy"])

from string import punctuation
punctuations = list(punctuation)

from scipy.stats import pearsonr

import seaborn as sns


emotions = ['anger', 'brain dysfunction (forget)', 'emptiness', 'hopelessness', 'loneliness', 'sadness', 'suicide intent', 'worthlessness']

def plot_by_weekday(dataset):

    weekday_dict = {}
    for item in dataset:
        date = item['date']
        #str = '2022-12-17 15:35:50'

        date = datetime.strptime(date, '%Y-%m-%d %H:%M:%S')
        weekday = date.weekday()
    
        for x in item['emotions']:
            label = x + '_' + str(weekday)
            if (label not in weekday_dict):
                weekday_dict[label]  = 0
            else:
                weekday_dict[label] += 1

    weekday_dict = dict(sorted(weekday_dict.items(), key=lambda item: item[0]))
    print('weekday_dict: ', weekday_dict)
    weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    values = []
    for k, v in weekday_dict.items():
        for feeling in emotions:
            if (k.split('_')[0]==feeling):
                values.append(v)
    sublists = [values[x:x+7] for x in range(0, len(values), 7)]


    X_axis = np.arange(len(weekdays)) 

    for feel, data, idx in zip(emotions, sublists, range(0, len(emotions))):
        print(feel, data, idx)
        plt.bar(X_axis + 0.08*idx, data, 0.08, label = feel) 
  
    plt.xticks(X_axis, weekdays) 
    plt.xlabel("Weekdays") 
    plt.ylabel("Frequencies") 
    plt.title("Emotions by frequencies in each weekday") 
    plt.legend() 
    plt.grid(color = 'gray', linestyle = '--', linewidth = 0.5)
    plt.tight_layout()
    plt.show() 

def plot_by_weekday_combined(dataset, emotion = 'suicide intent'):

    weekday_dict_all = {}
    weekday_dict = {}

    for item in dataset:
        date = item['date']
        #str = '2022-12-17 15:35:50'

        date = datetime.strptime(date, '%Y-%m-%d %H:%M:%S')
        weekday = date.weekday()

        label = str(weekday)
        if (label not in weekday_dict_all):
            weekday_dict_all[label] = 0
        else:
            weekday_dict_all[label] += len(item['emotions'])
    
    for item in dataset:
        date = item['date']
        #str = '2022-12-17 15:35:50'

        date = datetime.strptime(date, '%Y-%m-%d %H:%M:%S')
        weekday = date.weekday()

        label = str(weekday)
        
        if (emotion in item['emotions']):
            
            if (label not in weekday_dict): weekday_dict[label]  = 0
            else: weekday_dict[label] += 1
            
    weekday_dict = dict(sorted(weekday_dict.items(), key=lambda item: item[0]))
    weekday_dict_all = dict(sorted(weekday_dict_all.items(), key=lambda item: item[0]))
    
    print('weekday_dict: ', weekday_dict)
    weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    values = []
    for k, v in weekday_dict.items():
        #for feeling in emotions:
            #if (k.split('_')[0]==feeling):
        values.append(v)
    
    for k, v in weekday_dict_all.items():
        values.append(v)
    
    sublists = [values[x:x+7] for x in range(0, len(values), 7)]


    X_axis = np.arange(len(weekdays)) 
    emo_list  = [emotion, 'all']
    
    for data, idx, emo in zip(sublists, range(0, len(emotions)), emo_list):
        print(data, idx)
        plt.bar(X_axis + 0.4*idx, data, 0.4, label = emo) 
    
    for i in range(len(sublists[0])):
        plt.text(i + 0, sublists[0][i] + 7, sublists[0][i], ha = 'center')
        
    for i in range(len(sublists[0])):
        plt.text(i + 0.4, sublists[1][i] + 7, sublists[1][i], ha = 'center')
        
    plt.xticks(X_axis, weekdays) 
    plt.xlabel("Weekdays") 
    plt.ylabel("Frequencies") 
    plt.title("Emotions by frequencies in weekdays") 
    plt.legend() 
    plt.grid(color = 'gray', linestyle = '--', linewidth = 0.5)
    plt.tight_layout()
    plt.show() 

def plot_by_24_hour(dataset):

    hour_dict = {}
    for item in dataset:
        date = item['date']
        #str = '2022-12-17 15:35:50'

        date = datetime.strptime(date, '%Y-%m-%d %H:%M:%S')
        hour = date.hour
    
        for x in item['emotions']:
            label = x + '_' + str(hour)
            if (label not in hour_dict):
                hour_dict[label]  = 0
            else:
                hour_dict[label] += 1

    hour_dict = dict(sorted(hour_dict.items(), key=lambda item: item[0]))
    print('hour_dict: ', hour_dict)
    hours = range(1, 25)
    values = []
    for k, v in hour_dict.items():
        for feeling in emotions:
            if (k.split('_')[0]==feeling):
                values.append(v)
    sublists = [values[x:x+24] for x in range(0, len(values), 24)]


    X_axis = np.arange(len(hours)) 

    for feel, data, idx in zip(emotions, sublists, range(0, len(emotions))):
        print(feel, data, idx)
        plt.bar(X_axis + 0.08*idx, data, 0.08, label = feel) 
       
    plt.xticks(X_axis, hours) 
    plt.xlabel("Hours") 
    plt.ylabel("Frequencies") 
    plt.title("Emotions by frequencies in each hour") 
    plt.legend() 
    plt.grid(color = 'gray', linestyle = '--', linewidth = 0.5)
    plt.tight_layout()
    plt.show() 

def plot_by_24_hour_emotion(dataset, emotion = 'suicide intent'):

    hour_dict = {}
    hour_dict_all = {}
    for item in dataset:
        date = item['date']
        date = datetime.strptime(date, '%Y-%m-%d %H:%M:%S')
        hour = date.hour
        label = str(hour)
       
        if (emotion in item['emotions']):
            
            if (label not in hour_dict): hour_dict[label]  = 0
            else: hour_dict[label] += 1

    for item in dataset:
        
        date = item['date']
        date = datetime.strptime(date, '%Y-%m-%d %H:%M:%S')
        hour = date.hour
        label = str(hour)
        
        if (label not in hour_dict_all): 
            hour_dict_all[label] = 0
        else: 
            hour_dict_all[label] += len(item['emotions'])

    hour_dict = dict(sorted(hour_dict.items(), key=lambda item: item[0]))
    #print('hour_dict: ', hour_dict)
    
    hour_dict_all = dict(sorted(hour_dict_all.items(), key=lambda item: item[0]))
    #print('hour_dict_all: ', hour_dict_all)
    
    
    
    hours = range(1, 25)
    values = []
    for k, v in hour_dict.items():
        values.append(v)
        
    for k, v in hour_dict_all.items():
        values.append(v)
        
    sublists = [values[x:x+24] for x in range(0, len(values), 24)]

    emo_list  = [emotion, 'all']
    X_axis = np.arange(len(hours)) 

    for data, idx, emo in zip(sublists, range(0, len(hours)), emo_list):
        print(data, idx)
        plt.bar(X_axis + 0.4*idx, data, 0.4, label = emo) 
    
    for i in range(len(sublists[0])):
        plt.text(i + 0, sublists[0][i] + 5, sublists[0][i], ha = 'center', fontsize = 'small')
        
    for i in range(len(sublists[0])):
        plt.text(i + 0.4, sublists[1][i] + 5, sublists[1][i], ha = 'center', fontsize = 'small')
    
    plt.xticks(X_axis, hours) 
    plt.xlabel("Hours") 
    plt.ylabel("Frequencies") 
    plt.title("Emotions by frequencies in each hour") 
    plt.legend() 
    plt.grid(color = 'gray', linestyle = '--', linewidth = 0.5)
    plt.tight_layout()
    plt.show() 
   
    
def plot_by_24_hour_combined(dataset):

    
    hour_dict = {}

    for item in dataset:
        date = item['date']
        #str = '2022-12-17 15:35:50'

        date = datetime.strptime(date, '%Y-%m-%d %H:%M:%S')
        hour = date.hour
        label = str(hour)
        
        if (label not in hour_dict):
            hour_dict[label] = 0
        else:
            hour_dict[label] += len(item['emotions'])

    hour_dict = dict(sorted(hour_dict.items(), key=lambda item: item[0]))
    print('hour_dict: ', hour_dict)
    hours = range(1, 25)
    values = []
    for k, v in hour_dict.items():
        values.append(v)
    sublists = [values[x:x+24] for x in range(0, len(values), 24)]
    X_axis = np.arange(len(hours)) 

    for data, idx in zip(sublists, range(0, len(hours))):
        print(data, idx)
        plt.bar(X_axis + 0.5*idx, data, 0.5, label = "all emotions") 
    
    for i in range(len(data)):
  
        plt.text(i, data[i] + 5, data[i], ha = 'center', fontsize = 'x-small')
    
    plt.xticks(X_axis, hours) 
    plt.xlabel("Hours") 
    plt.ylabel("Frequencies") 
    plt.title("Emotions by frequencies in each hour") 
    plt.legend() 
    plt.grid(color = 'gray', linestyle = '--', linewidth = 0.5)
    plt.tight_layout()
    plt.show() 

def plot_by_emotion(dataset):

    
    emotion_dict = {}

    for item in dataset:
        
        for emotion in item['emotions']:
        
            if (emotion not in emotion_dict):
                emotion_dict[emotion]  = 0
            else:
                emotion_dict[emotion] += 1

    emotion_dict = dict(sorted(emotion_dict.items(), key=lambda item: item[1], reverse = True))
    print('emotion_dict: ', emotion_dict)
    
    removed_feels = ['lack of energy', 'self hate']
    y = [v for k, v in emotion_dict.items() if k not in removed_feels]
    x = [k for k, v in emotion_dict.items() if k not in removed_feels]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(y, labels=x, autopct='%.1f%%', wedgeprops={'linewidth': 2.0, 'edgecolor': 'white'}, startangle = 0)
    ax.set_title('Emotions by frequencies')
    plt.tight_layout()
    plt.show() 

def plot_by_text_length(dataset):

    result_dict = {}
    for item in dataset:
        doc = nlp(item['text'])
        if (len(doc) not in result_dict):
            result_dict[len(doc)] = 1
        else:
            result_dict[len(doc)] += 1

    result_dict = dict(sorted(result_dict.items(), key=lambda item: item[0], reverse = True))
    print('result_dict: ', result_dict)
    
    data = [v for k, v in result_dict.items()]
    labels = [k for k, v in result_dict.items()]
    plt.bar(labels, data) 
  
    #plt.xticks(X_axis, labels) 
    plt.xlabel("Text lengths") 
    plt.ylabel("Frequencies") 
    #plt.title("Text lengths by frequencies") 
    #plt.legend() 
    plt.grid(color = 'gray', linestyle = '--', linewidth = 0.5)
    plt.tight_layout()
    plt.show() 


def get_vocab_size(dataset):

    # 15410
    vocab_dict = {}
    len_list = []
    for item in dataset:
        doc = nlp(item['text'])
        temp_len = len([token for token in doc if not token.is_punct])
        
        len_list.append(temp_len)
        
        for token in doc:
            if (token.text.lower() not in vocab_dict):
                if not token.is_punct: # remove punctuations
                    vocab_dict[token.text.lower()] = 1
            else:
                vocab_dict[token.text.lower()] += 1
    
    print('avg len: ', sum(len_list)/len(len_list))
    return vocab_dict

def get_sncdl_vocab():
        
    # vocab: 11869, 2 classes, reddit
    df = pd.read_csv('dataset/sncdl/combined-set.csv')
    vocab_dict = {}
    len_list = []
    
    for index, row in df.iterrows():
    
        text = str(row['title_clean']) + ' ' + str(row['selftext_clean']) + ' ' + str(row['megatext_clean'])
        temp_len = len((str(row['title_clean']) + ' ' + str(row['selftext_clean'])).strip().split())
        len_list.append(temp_len)
        
        for word in text.split():
         
            if (word.strip() != ''):
                if (word not in vocab_dict):
                    vocab_dict[word] = 1
                else:
                    vocab_dict[word] += 1
    print('avg len: ', sum(len_list)/len(len_list))
    return vocab_dict 

def get_mde_vocab():
    
    # vocab: 12647, 3 classes, tweet
    dataset = read_list_from_json_file('dataset/mde/dataset.json')
    vocab_dict = {}
    len_list = []
    for item in dataset:
        temp_len = len(item['text'].strip().split())
        len_list.append(temp_len)
        for word in item['text'].split():
            if (word.strip() != ''):
                if (word not in vocab_dict):
                    vocab_dict[word] = 1
                else:
                    vocab_dict[word] += 1
    
    print('avg len: ', sum(len_list)/len(len_list))
    return vocab_dict 


def get_bldc_vocab():
    # vocab: , 2 classes, tweet
    dataset = read_list_from_json_file('dataset/bldc/dataset.json')
    vocab_dict = {}
    
    len_list = []
    
    for item in dataset['Sheet1']:
        
        temp_len = len(item['text'].strip().split())
        len_list.append(temp_len)
        for word in item['text'].split():
            if (word.strip() != ''):
                if (word not in vocab_dict):
                    vocab_dict[word] = 1
                else:
                    vocab_dict[word] += 1
    
    print('avg len: ', sum(len_list)/len(len_list))
    return vocab_dict 

def get_gometions_vocab():

   
    # vocab: xxx, 27 classes, reddit
    df = pd.read_csv('dataset/goemotions/goemotions_all.csv')
    vocab_dict = {}
    len_list = []
    
    for index, row in df.iterrows():
    
        text = str(row['text'])
        temp_len = len(text.split())
        len_list.append(temp_len)
        
        for word in text.split():
         
            if (word.strip() != ''):
                if (word not in vocab_dict):
                    vocab_dict[word] = 1
                else:
                    vocab_dict[word] += 1
    print('avg len: ', sum(len_list)/len(len_list))
    return vocab_dict 
    
# https://datasetsearch.research.google.com/search?src=3&query=depression&docid=L2cvMTFrNHAzMjN6bQ%3D%3D
# https://arxiv.org/pdf/2005.00547.pdf


def keyword_by_emotion(dataset, top_n = 15):
    
    emo_dict = {}
    for item in dataset:
        doc = nlp(item['text'])
        for token in doc:
            if(token.pos_ != 'ADJ'): continue
            word = token.text.lower().strip()
            if (word not in stop_words and word not in punctuations):
                for emo in item['emotions']:
                    label = emo + '_' + word
                    if (label not in emo_dict): emo_dict[label] = 0
                    else: emo_dict[label] += 1
    
    emo_dict = dict(sorted(emo_dict.items(), key=lambda item: item[1], reverse = True))
    print(emo_dict)   

   
    label_list = []
    for emo in emotions:
        i = 0
        temp_list = []
        #print(emo)
        for k, v in emo_dict.items():
            if (emo == k.split('_')[0]):
                #print(k.split('_')[1], '(' + str(v) + ')')
                temp_list.append(k.split('_')[1] + ' (' + str(v) + ')')
                i += 1
            if (i == top_n): 
                label_list.append(temp_list)
                break
        
    
    for x, y, z, t in zip(label_list[0], label_list[1], label_list[2], label_list[3]):
        print(x + ' & ' + y + ' & ' + z + ' & ' + t + ' \\\\')
    
    print('---------------------------------------') 
    print('---------------------------------------') 
    for x, y, z, t in zip(label_list[4], label_list[5], label_list[6], label_list[7]):
        print(x + ' & ' + y + ' & ' + z + ' & ' + t + ' \\\\')


def show_heat_map(dataset):

    heat_dict = {}
    for emo1 in emotions:
        for emo2 in emotions:
            if (emo1 == emo2):
                heat_dict[emo1 + '_' + emo2] = 1
            
            else:
                # create distributions
                dis1 = []
                dis2 = []
                for item in dataset:
                    if (emo1 in item['emotions']): dis1.append(1)
                    else: dis1.append(0)
            
                    if (emo2 in item['emotions']): dis2.append(1)
                    else: dis2.append(0)
            
                # pearson correlation
                corr, _ = pearsonr(dis1, dis2)
                heat_dict[emo1 + '_' + emo2] = corr
    
    print(heat_dict)
    
    heat_values = []
    for k, v in heat_dict.items():
        heat_values.append(v)
    
    heat_values = np.reshape(heat_values, (len(emotions), len(emotions)))
        
    print(heat_values)
    
    # Create a dataset
    df = pd.DataFrame(heat_values, columns=emotions)

    plt.figure(figsize=(8,8))
    plt.rcParams.update({'font.size': 12})
    
    emotions_sub = ['anger', 'brain dys.', 'emptiness', 'hopelessness', 'loneliness', 'sadness', 'suicide', 'worthlessness']
    
    xlabels = [x for x in emotions_sub]
    ylabels = xlabels
    
    heat_map = sns.heatmap(df, xticklabels=xlabels, yticklabels=ylabels, linewidth = 1, cmap="YlGnBu", annot = True)
    plt.title("")
    plt.tight_layout()
    plt.show()

def emotion_distribution_by_subset(train_set, val_set, test_set):

    train_dict, val_dict, test_dict = {}, {}, {}
    
    for item in train_set:
        for e in item['emotions']:
            if (e not in train_dict):
                train_dict[e] = 1
            else:
                train_dict[e] += 1
                
    for item in val_set:
        for e in item['emotions']:
            if (e not in val_dict):
                val_dict[e] = 1
            else:
                val_dict[e] += 1
            
    for item in test_set:
        for e in item['emotions']:
            if (e not in test_dict):
                test_dict[e] = 1
            else:
                test_dict[e] += 1
    
    train_dict = dict(sorted(train_dict.items(), key=lambda item: item[0]))
    val_dict = dict(sorted(val_dict.items(), key=lambda item: item[0]))
    test_dict = dict(sorted(test_dict.items(), key=lambda item: item[0]))
    
    print('train_dict: ', train_dict)
    print('val_dict: ', val_dict)
    print('test_dict: ', test_dict)
    
    values1, values2, values3 = [], [], []
    
    values11, values22, values33 = [], [], []
    
    
    for k, v in train_dict.items(): 
        values1.append(round((v/sum(train_dict.values()))*100, 2))
        values11.append(v)
    for k, v in val_dict.items(): 
        values2.append(round((v/sum(val_dict.values()))*100, 2))
        values22.append(v)
    for k, v in test_dict.items(): 
        values3.append(round((v/sum(test_dict.values()))*100, 2))
        values33.append(v)
    
    
    emo_dict = {}
    for x, y, z, e in zip(values1, values2, values3, emotions):
        emo_dict[e] = [x, y, z]
    
    emo_dict2 = {}
    for x, y, z, e in zip(values11, values22, values33, emotions):
        emo_dict2[e] = [x, y, z]
    
    print(emo_dict)
    sets = ['train', 'val', 'test']
    
    fig, ax = plt.subplots()

    # stacked bar chart
    
    ax.barh(sets, emo_dict['anger'], label = 'anger')
    
    add_list = emo_dict['anger']

    prev = 'anger'
    for e in emotions[1:]:
        print('add_list: ', add_list)
        ax.barh(sets, emo_dict[e], left = add_list, label =  e, alpha = 0.8)
        prev = e
        add_list = [x + y for x, y in zip(add_list, emo_dict[e])]

    # labels
    i = 0
    for bar in ax.patches:
        print(i)
        j = i%3
        k = int(i/3)
        print('--', j)
        print('--', k)
        #emo_dict2[emotions[k]][j]
        emo_text = ''
        if (emotions[k] == 'brain dysfunction (forget)'):
        
            emo_text = 'brain dysfunction'
        else:
            emo_text = emotions[k]
        if (bar.get_width() == 0): continue
        ax.text(bar.get_x() + bar.get_width()*0.7, bar.get_height()*0.5 + bar.get_y(), emo_text + '\n' + str(round(bar.get_width(),2)) + '% \n' + str(emo_dict2[emotions[k]][j]) + ' examples', ha = 'center', color = 'black', size = 10, alpha = 0.6, rotation = 60, 
         rotation_mode = 'anchor')
        i = i + 1


    #ax.legend(loc = 'lower right')
    #ax.set_xlabel('Number of quotations')
    plt.tight_layout()    
    plt.show()

#..................................................................
if __name__ == "__main__":

    #print(len(get_mde_vocab()))
    #print(len(get_sncdl_vocab()))
    #print(len(get_bldc_vocab()))
    #print(len(get_gometions_vocab()))
    
    dataset = read_list_from_jsonl_file('dataset/train.json')
    dataset += read_list_from_jsonl_file('dataset/val.json')
    dataset += read_list_from_jsonl_file('dataset/test.json')
    
    show_heat_map(dataset)
    #keyword_by_emotion(dataset)
    #plot_by_24_hour_emotion(dataset)
    #plot_by_24_hour_emotion(dataset)
    #plot_by_weekday(dataset)
    #plot_by_weekday_combined(dataset)
    #plot_by_24_hour(dataset)
    #plot_by_24_hour_combined(dataset)
    #plot_by_emotion(dataset)
    #plot_by_text_length(dataset)
    #print(len(get_vocab_size(dataset)))
    
    
    train_set = read_list_from_jsonl_file('dataset/train.json')
    val_set = read_list_from_jsonl_file('dataset/val.json')
    test_set = read_list_from_jsonl_file('dataset/test.json')
    
    emotion_distribution_by_subset(train_set, val_set, test_set)