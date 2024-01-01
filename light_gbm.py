import pandas as pd
import numpy as np

import lightgbm as lgbm
from lightgbm import LGBMClassifier, LGBMRegressor

from file_io import *

emotion_list = ['anger', 'brain dysfunction (forget)', 'emptiness', 'hopelessness', 'loneliness', 'sadness', 'suicide intent', 'worthlessness']

def convert_labels(labels):
    
    labels2 = []
    for idx, label in enumerate(labels):
        try: 
            label = str(label[0])
        except:
            label = str(label)
            pass

        temp = []
        num = len(emotion_list) - len(label)
        if (num == 0): 
            temp = [int(x) for x in label]
        else:
            temp = ''.join(['0']*num) + str(label)   
            temp = [int(x) for x in temp]
            
        labels2.append(temp)
        
    return labels2
    
    
train_set = read_list_from_jsonl_file('dataset/train.json')
val_set = read_list_from_jsonl_file('dataset/val.json')
test_set = read_list_from_jsonl_file('dataset/test.json')
train_set = train_set  + test_set + val_set

data_labels = [item['label_id'] for item in train_set]
df_labels = pd.DataFrame(data_labels, columns=['labels'])
print('df_labels: ', df_labels)

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re

#from transformers import AutoTokenizer

# Download necessary NLTK corpora
nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemma = WordNetLemmatizer()

#tokenizer = AutoTokenizer.from_pretrained('facebook/bart-base')

def clean_text(text):
    
    '''text = tokenizer.encode_plus(text, max_length=256, add_special_tokens=True, return_token_type_ids=False, padding = "max_length", return_attention_mask=True, return_tensors='pt')
    text = ' '.join(str(x) for x in text['input_ids'].tolist()[0])'''
    
    '''text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^\w\s]', '', text, re.UNICODE) # remove punctuation
    
    # cleaning everything except alphabetical and numerical characters
    text = re.sub("[^a-zA-Z0-9]"," ",text)
    
    # tokenizing and lemmatizing
    text = nltk.word_tokenize(text.lower())
    text = [lemma.lemmatize(word) for word in text]
    
    # removing stopwords
    text = [word for word in text if word not in stop_words]
    
    # joining
    text = " ".join(text)'''
    
    return text

for item in train_set: item['text'] = clean_text(item['text'])

for i in range(0,5): print(train_set[i]['text'],end="\n\n")
    
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()
#vectorizer = TfidfVectorizer(max_features=5000)
data_features = vectorizer.fit_transform([item['text'] for item in train_set])
data_features = data_features.toarray()

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(data_features, np.asarray(df_labels), test_size=0.3, train_size = 0.7, random_state=0)
x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=0.5, random_state=0)

print(x_train.shape)
print(y_train.shape)

print(x_val.shape)
print(y_val.shape)

print(x_test.shape)
print(y_test.shape)

model = lgbm.LGBMClassifier(objective='multiclass', learning_rate = 0.5, verbose = -1)

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_train = le.fit_transform(y_train.ravel())

model.fit(x_train, y_train)

predictions = model.predict(x_test)
predictions = le.inverse_transform(predictions)
predictions = convert_labels(predictions)
print('predictions: ', predictions)

y_test = convert_labels(y_test)
print('y_test: ', y_test)

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

f1_mi = f1_score(y_true=y_test, y_pred=predictions, average='micro')
re_mi = recall_score(y_true=y_test, y_pred=predictions, average='micro')
pre_mi = precision_score(y_true=y_test, y_pred=predictions, average='micro')
    
f1_mac = f1_score(y_true=y_test, y_pred=predictions, average='macro')
re_mac = recall_score(y_true=y_test, y_pred=predictions, average='macro')
pre_mac = precision_score(y_true=y_test, y_pred=predictions, average='macro')
    
result = {}
result['f1_micro'] = f1_mi
result['recall_micro'] = re_mi
result['precision_micro'] = pre_mi
    
result['f1_macro'] = f1_mac
result['recall_macro'] = re_mac
result['precision_macro'] = pre_mac

print(result)