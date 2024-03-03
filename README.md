# DepressionEmo
"DepressionEmo: A novel dataset for multilabel classification of depression emotions"
https://arxiv.org/pdf/2401.04655.pdf

# Dataset
## Subsets
The dataset is divided into 3 subsets:
* Training set
* Validation set
* Test set

## An example data
Each example contains "id", "title", "post", "text", "upvotes", "date", "emotions", and "label_id". We use "text" (concatenate from "title" and "post") for the depression detection.

```
{"id": "zq1lwl", "title": "goodbye.", "post": "I'm done. I have a bottle of jack danials and couple bottles of sleeping meds waiting for me when I get home. \n\nI'm excited to leave this place. There will be no regret, no loneliness, no sorrow, and no misery. I am looking forward to the peace of nonexistence.\n\nI love you all.", "text": "goodbye. ### I'm done. I have a bottle of jack danials and couple bottles of sleeping meds waiting for me when I get home. I'm excited to leave this place. There will be no regret, no loneliness, no sorrow, and no misery. I am looking forward to the peace of nonexistence. I love you all.", "upvotes": 102, "date": "2022-12-19 19:50:52", "emotions": ["emptiness", "hopelessness"], "label_id": 110000}
```

There are 8 depression emotions:
```
emotion_list = ['anger', 'brain dysfunction (forget)', 'emptiness', 'hopelessness', 'loneliness', 'sadness', 'suicide intent', 'worthlessness']
```

The "label_id" field is a number, whose each digit represents a state of with emotion (1) and without emotion (0). For example:
* ["emptiness", "hopelessness"] ->  00110000 -> 110000
* ["anger"] -> 10000000
* ['anger', 'brain dysfunction (forget)', 'emptiness', 'hopelessness', 'loneliness', 'sadness', 'suicide intent', 'worthlessness'] -> 11111111

# Training
This is a multilabel classification problem, and we use only 1 model to detect all emotions at once.

## SVM, Light GBM, and XGBoost
All these methods use TfidfVectorizer and have no preprocessing steps. To train the models by these methods use:

```
python svm.py
python xgb.py
python light_gbm.py
```

## BERT
To train the model:
```
python bert.py  --mode "train" --model_name "bert-base-cased" --epochs 25 --batch_size 8 --max_length 256 --train_path "Dataset/train.json" --val_path "Dataset/val.json" --test_path "Dataset/test.json"
```

To test the model:
```
python bert.py --mode "test" --train_path "Dataset/train.json" --val_path "Dataset/val.json" --test_path "Dataset/test.json" --max_length 256 --test_batch_size 16
```
Note that you must give all sets (including the test set) so that BERT can capture all class names (categories).

## GAN BERT
GAN-BERT uses ''Dataset/label_names.json'' to capture all categories, including category "unlabelled".
To train the model:

```
python gan.py --mode "train" --model_name "bert-base-cased" --lr_discriminator 2e-5 --lr_generator 2e-5 --epochs 25 --batch_size 8
```

To test the model:

```
python gan.py --mode "test" --model_path "model_bert-base-cased.bin"  --test_file "Dataset/test.json"
```

## BART
To train the model:

```
python seq2seq.py --mode "train" --model_name "facebook/bart-base" --train_path "Dataset/train.json" --val_path "Dataset/val.json" --test_path "Dataset/test.json" --epochs 25 --batch_size 4 --max_source_length 256
```

To test the model:

```
python seq2seq.py --mode "test" --model_name "facebook/bart-base" --model_path "bart-base\model_checkpoint_xxx" --test_path "Dataset/test.json" --test_batch_size 4 --max_source_length 256 --min_target_length 1
```

# Contact
Please feel free to ask or discuss with us.
* Email: abubakarsiddiqurra@unomaha.edu
* Email: tahoangthang@gmail.com

