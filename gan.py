import torch
import io
import torch.nn.functional as F
import random
import numpy as np
import time
import math
import datetime
import torch.nn as nn
from transformers import *
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

from preprocessing import *
from file_io import *
import datasets
import argparse

emotion_list = ['anger', 'brain dysfunction (forget)', 'emptiness', 'hopelessness', 'loneliness', 'sadness', 'suicide intent', 'worthlessness']

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
    
def get_qc_examples(input_file):
    
    examples = []
    dataset = datasets.load_dataset('json', data_files = input_file, split="train")
    for item in dataset:
        text = '[CLS] ' + item['text'] + ' [SEP]'
        label = ''
        try:
            label = str(item['label_id'])
        except:
            label = 'unlabelled'     
        examples.append((text, label))
        
    
    return examples

def generate_data_loader(tokenizer, max_seq_length, batch_size, input_examples, label_masks, label_map, do_shuffle = False, balance_label_examples = False):
    '''
        Generate a Dataloader given the input examples, eventually masked if they are 
        to be considered NOT labeled.
    '''
    examples = []
    
    # Count the percentage of labeled examples  
    num_labeled_examples = 0
    for label_mask in label_masks:
        if label_mask: 
            num_labeled_examples += 1
    label_mask_rate = num_labeled_examples/len(input_examples)
    
    #print('label_mask_rate: ', label_mask_rate)

    # if required it applies the balance
    for index, ex in enumerate(input_examples): 
        if label_mask_rate == 1 or not balance_label_examples:
            examples.append((ex, label_masks[index]))
        else:
            # IT SIMULATE A LABELED EXAMPLE
            if label_masks[index]:
                balance = int(1/label_mask_rate)
                balance = int(math.log(balance,2))
                if balance < 1: balance = 1
                for b in range(0, int(balance)):
                    examples.append((ex, label_masks[index]))
            else:
                examples.append((ex, label_masks[index]))
  
    #-----------------------------------------------
    # Generate input examples to the Transformer
    #-----------------------------------------------
    input_ids = []
    input_mask_array = []
    label_mask_array = []
    label_id_array = []

    # Tokenization 
    for (text, label_mask) in examples:
        encoded_sent = tokenizer.encode(text[0], add_special_tokens=True, max_length=max_seq_length, padding="max_length", truncation=True)
        input_ids.append(encoded_sent)
        label_id_array.append(label_map[text[1]])
        label_mask_array.append(label_mask)
  
    # Attention to token (to ignore padded input wordpieces)
    for sent in input_ids:
        att_mask = [int(token_id > 0) for token_id in sent]                          
        input_mask_array.append(att_mask)
    
    # Convertion to Tensor
    input_ids = torch.tensor(input_ids) 
    input_mask_array = torch.tensor(input_mask_array)
    label_id_array = torch.tensor(label_id_array, dtype=torch.long)
    label_mask_array = torch.tensor(label_mask_array)

    # Building the TensorDataset
    dataset = TensorDataset(input_ids, input_mask_array, label_id_array, label_mask_array)

    if do_shuffle:
        sampler = RandomSampler
    else:
        sampler = SequentialSampler

    # Building the DataLoader
    return DataLoader(
              dataset,  # The training samples.
              sampler = sampler(dataset), 
              batch_size = batch_size) # Trains with this batch size.

def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))
    
#------------------------------
#   The Generator as in 
#   https://www.aclweb.org/anthology/2020.acl-main.191/
#   https://github.com/crux82/ganbert
#------------------------------
class Generator(nn.Module):
    def __init__(self, noise_size=768, output_size=768, hidden_sizes=[768], dropout_rate=0.1):
        super(Generator, self).__init__()
        layers = []
        hidden_sizes = [noise_size] + hidden_sizes
        for i in range(len(hidden_sizes)-1):
            layers.extend([nn.Linear(hidden_sizes[i], hidden_sizes[i+1]), nn.LeakyReLU(0.2, inplace=True), nn.Dropout(dropout_rate)])

        layers.append(nn.Linear(hidden_sizes[-1],output_size))
        self.layers = nn.Sequential(*layers)

    def forward(self, noise):
        output_rep = self.layers(noise)
        return output_rep

#------------------------------
#   The Discriminator
#   https://www.aclweb.org/anthology/2020.acl-main.191/
#   https://github.com/crux82/ganbert
#------------------------------
class Discriminator(nn.Module):
    def __init__(self, input_size=768, hidden_sizes=[768], num_labels=2, dropout_rate=0.1):
        super(Discriminator, self).__init__()
        self.input_dropout = nn.Dropout(p=dropout_rate)
        layers = []
        hidden_sizes = [input_size] + hidden_sizes
        for i in range(len(hidden_sizes)-1):
            layers.extend([nn.Linear(hidden_sizes[i], hidden_sizes[i+1]), nn.LeakyReLU(0.2, inplace=True), nn.Dropout(dropout_rate)])

        self.layers = nn.Sequential(*layers) #per il flatten
        self.logit = nn.Linear(hidden_sizes[-1], num_labels+1) # +1 for the probability of this sample being fake/real.
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, input_rep):
        input_rep = self.input_dropout(input_rep)
        last_rep = self.layers(input_rep)
        logits = self.logit(last_rep)
        probs = self.softmax(logits)
        return last_rep, logits, probs

def make_noise(input_list, noise_rate = 1):

    #print('len(input_list): ', len(input_list))
    times = int(len(input_list)*noise_rate)
    pos_list = []

    while(True):
        if  (times == 0): break
        pos = random.randint(0, len(input_list) - 1)
        if (pos not in pos_list):
            input_list[pos] = random.uniform(-1, 1)
            pos_list.append(pos)
            times = times - 1

    return input_list

def train_model(model_name = '',
                learning_rate_discriminator = 5e-7, learning_rate_generator = 5e-7, num_train_epochs = 5, batch_size = 16):

    seed_val = 42
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_val)
    
    # GPU or CPU
    if torch.cuda.is_available():    
        device = torch.device("cuda")
        print('There are %d GPU(s) available.' % torch.cuda.device_count())
        print('We will use the GPU:', torch.cuda.get_device_name(0))
    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")
    
    #--------------------------------
    #  Transformer parameters
    #--------------------------------
    max_seq_length = 256
    #batch_size = 2

    #--------------------------------
    #  GAN-BERT specific parameters
    #--------------------------------
    # number of hidden layers in the generator, 
    # each of the size of the output space
    num_hidden_layers_g = 2
    # number of hidden layers in the discriminator, 
    # each of the size of the input space
    num_hidden_layers_d = 2
    # size of the generator's input noisy vectors
    noise_size = 768
    # dropout to be applied to discriminator's input vectors
    out_dropout_rate = 0.5 

    # Replicate labeled data to balance poorly represented datasets, 
    # e.g., less than 1% of labeled material
    apply_balance = False

    #--------------------------------
    #  Optimization parameters
    #--------------------------------
    #learning_rate_discriminator = 5e-7
    #learning_rate_generator = 5e-7
    epsilon = 2e-7
    #num_train_epochs = 5
    multi_gpu = True
    # Scheduler
    apply_scheduler = False
    warmup_proportion = 0.1
    # Print
    print_each_n_step = 100

    #--------------------------------
    #  Adopted Tranformer model
    #--------------------------------
    # Since this version is compatible with Huggingface transformers, you can uncomment
    # (or add) transformer models compatible with GAN

    #model_name = "bert-base-cased"
    #model_name = "bert-base-uncased"
    #model_name = "roberta-base"
    #model_name = "albert-base-v2"
    #model_name = "xlm-roberta-base"

    #--------------------------------
    #  Retrieve dataset
    #--------------------------------

    labeled_file = "dataset/train.json" 
    unlabeled_file = "dataset/test.json" 
    val_filename = "dataset/val.json" 
    test_filename = "dataset/test.json" 
                      
    transformer = AutoModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # load the examples
    dataset = get_qc_examples(labeled_file)
    #random.shuffle(dataset)
    
    labeled_examples = dataset
    val_examples = get_qc_examples(val_filename)
    test_examples = get_qc_examples(test_filename)

    #unlabeled_examples = get_qc_examples(unlabeled_file)
    unlabeled_examples = ()

    print('labeled_examples: ', len(labeled_examples))
    print('val_examples: ', len(val_examples))
    print('test_examples: ', len(test_examples))
    print('unlabeled_examples: ', len(unlabeled_examples))
    
    #label_list = ['unlabelled']
    '''for item in labeled_examples + val_examples + test_examples:
        label_list.append(item[1])
    
    label_list = list(set(label_list))
    print('label_list: ', label_list)'''
    
    label_list = read_list_from_json_file('dataset/label_names.json')
    print('label_list: ', label_list)
    
    label_map = {}
    for (i, label) in enumerate(label_list):
        label_map[label] = i

    print('label_map: ', label_map)
    #------------------------------
    #   Load the train dataset
    #------------------------------
    train_examples = labeled_examples # add validation to train set also
    random.shuffle(train_examples)
    #The labeled (train) dataset is assigned with a mask set to True
    train_label_masks = np.ones(len(train_examples), dtype=bool)
    #If unlabel examples are available
    
    # try not use this
    if unlabeled_examples:
        train_examples += unlabeled_examples
        #The unlabeled (train) dataset is assigned with a mask set to False
        tmp_masks = np.zeros(len(unlabeled_examples), dtype=bool)
        train_label_masks = np.concatenate([train_label_masks, tmp_masks])

    #print('label_map: ', label_map)
    train_dataloader = generate_data_loader(tokenizer, max_seq_length, batch_size, train_examples, train_label_masks, label_map, do_shuffle = True, balance_label_examples = apply_balance)

    print('train_dataloader: ', len(train_dataloader))
    #------------------------------
    #   Load the test dataset
    #------------------------------
    # The labeled (test) dataset is assigned with a mask set to True
    test_label_masks = np.ones(len(test_examples), dtype=bool)
    test_dataloader = generate_data_loader(tokenizer, max_seq_length, batch_size, test_examples, test_label_masks, label_map, 
                        do_shuffle = False, balance_label_examples = False)
    print('test_dataloader: ', len(test_dataloader))
    
    #------------------------------
    #   Load the validation dataset
    #------------------------------
    val_label_masks = np.ones(len(val_examples), dtype=bool)
    val_dataloader = generate_data_loader(tokenizer, max_seq_length, batch_size, val_examples, val_label_masks, label_map, do_shuffle = False, balance_label_examples = False)
    print('val_dataloader: ', len(val_dataloader))

    # The config file is required to get the dimension of the vector produced by 
    # the underlying transformer
    config = AutoConfig.from_pretrained(model_name)
    hidden_size = int(config.hidden_size)
    # Define the number and width of hidden layers
    hidden_levels_g = [hidden_size for i in range(0, num_hidden_layers_g)]
    hidden_levels_d = [hidden_size for i in range(0, num_hidden_layers_d)]

    #-------------------------------------------------
    #   Instantiate the Generator and Discriminator
    #-------------------------------------------------
    generator = Generator(noise_size=noise_size, output_size=hidden_size, hidden_sizes=hidden_levels_g, dropout_rate=out_dropout_rate)
    discriminator = Discriminator(input_size=hidden_size, hidden_sizes=hidden_levels_d, num_labels=len(label_list), dropout_rate=out_dropout_rate)

    # Put everything in the GPU if available
    if torch.cuda.is_available():    
        generator.cuda()
        discriminator.cuda()
        transformer.cuda()
        if multi_gpu:
            transformer = torch.nn.DataParallel(transformer)

    # print(config)
    training_stats = []

    # Measure the total training time for the whole run.
    total_t0 = time.time()

    #models parameters
    transformer_vars = [i for i in transformer.parameters()]
    d_vars = transformer_vars + [v for v in discriminator.parameters()]
    g_vars = [v for v in generator.parameters()]

    #optimizer
    dis_optimizer = torch.optim.AdamW(d_vars, lr=learning_rate_discriminator)
    gen_optimizer = torch.optim.AdamW(g_vars, lr=learning_rate_generator) 

    #scheduler
    if apply_scheduler:
        num_train_examples = len(train_examples)
        num_train_steps = int(num_train_examples / batch_size * num_train_epochs)
        num_warmup_steps = int(num_train_steps * warmup_proportion)

        scheduler_d = get_constant_schedule_with_warmup(dis_optimizer, num_warmup_steps = num_warmup_steps)
        scheduler_g = get_constant_schedule_with_warmup(gen_optimizer, num_warmup_steps = num_warmup_steps)

    # For each epoch...
    best_train_metric = 0
    best_epoch = -1
    
    #noise_rate = 0.1
    #noise_rate_minimum = 0.05
    for epoch_i in range(0, num_train_epochs):
        # ========================================
        #               Training
        # ========================================
        # Perform one full pass over the training set.
        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, num_train_epochs))
        print('Training...')

        # Measure how long the training epoch takes.
        t0 = time.time()

        # Reset the total loss for this epoch.
        tr_g_loss = 0
        tr_d_loss = 0

        # Put the model into training mode.
        transformer.train() 
        generator.train()
        discriminator.train()

        # For each batch of training data...
        for step, batch in enumerate(train_dataloader):

            # Progress update every print_each_n_step batches.
            if step % print_each_n_step == 0 and not step == 0:
                # Calculate elapsed time in minutes.
                elapsed = format_time(time.time() - t0)
                
                # Report progress.
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

            # Unpack this training batch from our dataloader. 
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)
            b_label_mask = batch[3].to(device)

            real_batch_size = b_input_ids.shape[0]
            
            # Encode real data in the Transformer
            model_outputs = transformer(b_input_ids, attention_mask=b_input_mask)
            hidden_states = model_outputs[-1]
             
            # Distort hidden_states as noises
            noises = hidden_states.detach().clone()
            modified_noises = []
            for noise in noises:
                modified_noises.append(make_noise(noise.tolist(), random.uniform(0.5, 1)))  
            modified_noises = torch.Tensor(modified_noises).to(device)
            
            # Generate fake data
            gen_rep = generator(modified_noises)

            # Generate the output of the Discriminator for real and fake data.
            # first, we put together the output of the transformer and the generator
            disciminator_input = torch.cat([hidden_states, gen_rep], dim=0)
            
            # Then, we select the output of the disciminator
            features, logits, probs = discriminator(disciminator_input)

            # Finally, we separate the discriminator's output for the real and fake
            # data
            features_list = torch.split(features, real_batch_size)
            D_real_features = features_list[0]
            #print('D_real_features: ', D_real_features)
            D_fake_features = features_list[1]
            #print('D_fake_features: ', D_fake_features)
          
            logits_list = torch.split(logits, real_batch_size)
            D_real_logits = logits_list[0]
            D_fake_logits = logits_list[1]
            #print('D_real_logits: ', D_real_logits)
            #print('D_fake_logits: ', D_fake_logits)
            
            probs_list = torch.split(probs, real_batch_size)
            D_real_probs = probs_list[0]
            D_fake_probs = probs_list[1]

            #---------------------------------
            #  LOSS evaluation
            #---------------------------------
            # generator's LOSS estimation
            g_loss_d = -1 * torch.mean(torch.log(1 - D_fake_probs[:,-1] + epsilon))            
            g_feat_reg = torch.mean(torch.pow(torch.mean(D_real_features, dim=0) - torch.mean(D_fake_features, dim=0), 2))
            g_loss = g_loss_d + g_feat_reg
      
            # disciminator's LOSS estimation
            logits = D_real_logits[:,0:-1]
            log_probs = F.log_softmax(logits, dim=-1)
            # The discriminator provides an output for labeled and unlabeled real data
            # so the loss evaluated for unlabeled data is ignored (masked)
            label2one_hot = torch.nn.functional.one_hot(b_labels, len(label_list))
            per_example_loss = -torch.sum(label2one_hot * log_probs, dim=-1)
            per_example_loss = torch.masked_select(per_example_loss, b_label_mask.to(device))
            labeled_example_count = per_example_loss.type(torch.float32).numel()

            # It may be the case that a batch does not contain labeled examples, 
            # so the "supervised loss" in this case is not evaluated
            if labeled_example_count == 0:
                D_L_Supervised = 0
            else:
                D_L_Supervised = torch.div(torch.sum(per_example_loss.to(device)), labeled_example_count)
                     
            D_L_unsupervised1U = -1 * torch.mean(torch.log(1 - D_real_probs[:, -1] + epsilon))
            D_L_unsupervised2U = -1 * torch.mean(torch.log(D_fake_probs[:, -1] + epsilon))
            
            d_loss = D_L_Supervised + D_L_unsupervised1U + D_L_unsupervised2U
            #d_loss = D_L_Supervised + D_L_unsupervised1U

            #---------------------------------
            #  OPTIMIZATION
            #---------------------------------
            # Avoid gradient accumulation
            gen_optimizer.zero_grad()
            dis_optimizer.zero_grad()

            # Calculate weigth updates
            # retain_graph=True is required since the underlying graph will be deleted after backward
            g_loss.backward(retain_graph=True)
            d_loss.backward() 
            
            # Apply modifications
            gen_optimizer.step()
            dis_optimizer.step()

            # A detail log of the individual losses
            #print("{0:.4f}\t{1:.4f}\t{2:.4f}\t{3:.4f}\t{4:.4f}".
            #      format(D_L_Supervised, D_L_unsupervised1U, D_L_unsupervised2U,
            #             g_loss_d, g_feat_reg))

            # Save the losses to print them later
            tr_g_loss += g_loss.item()
            tr_d_loss += d_loss.item()

            # Update the learning rate with the scheduler
            if apply_scheduler:
                scheduler_d.step()
                scheduler_g.step()
        
        # Calculate the average loss over all of the batches.
        avg_train_loss_g = tr_g_loss / len(train_dataloader)
        avg_train_loss_d = tr_d_loss / len(train_dataloader)             
        
        # Measure how long this epoch took.
        training_time = format_time(time.time() - t0)

        print("")
        print("  Average training loss generetor: {0:.3f}".format(avg_train_loss_g))
        print("  Average training loss discriminator: {0:.3f}".format(avg_train_loss_d))
        print("  Training epoch took: {:}".format(training_time))
            
        # ========================================
        #     TEST ON THE EVALUATION DATASET
        # ========================================
        # After the completion of each training epoch, measure our performance on
        # our test set.
        print("")
        print("Running Test set...")

        t0 = time.time()

        # Put the model in evaluation mode--the dropout layers behave differently
        # during evaluation.
        transformer.eval() 
        discriminator.eval()
        generator.eval()

        # Tracking variables for test set
        total_test_loss = 0
        all_preds = []
        all_labels_ids = []
        nll_loss = torch.nn.CrossEntropyLoss(ignore_index=-1) #loss

        # Evaluate data for one epoch
        for batch in test_dataloader:
            
            # Unpack this training batch from our dataloader. 
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)
            
            # Tell pytorch not to bother with constructing the compute graph during
            # the forward pass, since this is only needed for backprop (training).
            with torch.no_grad():        
                model_outputs = transformer(b_input_ids, attention_mask=b_input_mask)
                hidden_states = model_outputs[-1]
                _, logits, probs = discriminator(hidden_states)
                ###log_probs = F.log_softmax(probs[:,1:], dim=-1)
                filtered_logits = logits[:,0:-1]
                # Accumulate the test loss.
                total_test_loss += nll_loss(filtered_logits, b_labels)
                
            # Accumulate the predictions and the input labels
            _, preds = torch.max(filtered_logits, 1)
            all_preds += preds.detach().cpu()
            all_labels_ids += b_labels.detach().cpu()

        # Report the final accuracy for this validation run.
        all_preds = torch.stack(all_preds).numpy()
        all_labels_ids = torch.stack(all_labels_ids).numpy()
        #test_accuracy = np.sum(all_preds == all_labels_ids) / len(all_preds)
        #print("  Test Accuracy: {0:.4f}".format(test_accuracy))

        # convert back to labels
        label_map2 = dict((v,k) for k,v in label_map.items())
        all_preds = [label_map2[x] for x in all_preds]
        all_preds = convert_labels(all_preds)
        
        all_labels_ids = [label_map2[x] for x in all_labels_ids]
        all_labels_ids = convert_labels(all_labels_ids)

        
        # evaluate metrics
        f1_mi = f1_score(y_true=all_labels_ids, y_pred=all_preds, average='micro')
        re_mi = recall_score(y_true=all_labels_ids, y_pred=all_preds, average='micro')
        pre_mi = precision_score(y_true=all_labels_ids, y_pred=all_preds, average='micro')
    
        f1_mac = f1_score(y_true=all_labels_ids, y_pred=all_preds, average='macro')
        re_mac = recall_score(y_true=all_labels_ids, y_pred=all_preds, average='macro')
        pre_mac = precision_score(y_true=all_labels_ids, y_pred=all_preds, average='macro')
    
        test_result = {}
        test_result['f1_micro'] = f1_mi
        test_result['recall_micro'] = re_mi
        test_result['precision_micro'] = pre_mi
    
        test_result['f1_macro'] = f1_mac
        test_result['recall_macro'] = re_mac
        test_result['precision_macro'] = pre_mac

        print(test_result)

        # Calculate the average loss over all of the batches.
        avg_test_loss = total_test_loss / len(test_dataloader)
        avg_test_loss = avg_test_loss.item()
        
        # Measure how long the validation run took.
        test_time = format_time(time.time() - t0)
        
        print("  Test loss: {0:.4f}".format(avg_test_loss))
        print("  Test took: {:}".format(test_time))

        # ========================================
        #     TEST ON THE TRAINING DATASET
        # ========================================
        # After the completion of each training epoch, measure our performance on
        # our training set.
        
        print("")
        print("Running Train set...")

        t1 = time.time()
        
        # Tracking variables for train set
        total_train_loss = 0
        all_train_preds = []
        all_train_labels_ids = []
        nll_train_loss = torch.nn.CrossEntropyLoss(ignore_index=-1) #loss

        # Evaluate data for one epoch
        for batch in train_dataloader:
            
            # Unpack this training batch from our dataloader. 
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)
            
            # Tell pytorch not to bother with constructing the compute graph during
            # the forward pass, since this is only needed for backprop (training).
            with torch.no_grad():        
                model_outputs = transformer(b_input_ids, attention_mask=b_input_mask)
                hidden_states = model_outputs[-1]
                _, logits, probs = discriminator(hidden_states)
                ###log_probs = F.log_softmax(probs[:,1:], dim=-1)
                filtered_logits = logits[:,0:-1]
                # Accumulate the test loss.
                total_train_loss += nll_train_loss(filtered_logits, b_labels)
                
            # Accumulate the predictions and the input labels
            _, preds = torch.max(filtered_logits, 1)
            all_train_preds += preds.detach().cpu()
            all_train_labels_ids += b_labels.detach().cpu()

        # Report the final accuracy for this validation run.
        all_train_preds = torch.stack(all_train_preds).numpy()
        all_train_labels_ids = torch.stack(all_train_labels_ids).numpy()
        
        #train_accuracy = np.sum(all_train_preds == all_train_labels_ids) / len(all_train_preds)
        #print("  Train Accuracy: {0:.3f}".format(train_accuracy))
               
        # covert back to labels
        label_map2 = dict((v,k) for k,v in label_map.items())
        all_train_preds = [label_map2[x] for x in all_train_preds]
        all_train_preds = convert_labels(all_train_preds)
        
        all_train_labels_ids = [label_map2[x] for x in all_train_labels_ids]
        all_train_labels_ids = convert_labels(all_train_labels_ids)

        # evaluate metrics
        f1_mi = f1_score(y_true=all_train_labels_ids, y_pred=all_train_preds, average='micro')
        re_mi = recall_score(y_true=all_train_labels_ids, y_pred=all_train_preds, average='micro')
        pre_mi = precision_score(y_true=all_train_labels_ids, y_pred=all_train_preds, average='micro')
    
        f1_mac = f1_score(y_true=all_train_labels_ids, y_pred=all_train_preds, average='macro')
        re_mac = recall_score(y_true=all_train_labels_ids, y_pred=all_train_preds, average='macro')
        pre_mac = precision_score(y_true=all_train_labels_ids, y_pred=all_train_preds, average='macro')
    
        train_result = {}
        train_result['f1_micro'] = f1_mi
        train_result['recall_micro'] = re_mi
        train_result['precision_micro'] = pre_mi
    
        train_result['f1_macro'] = f1_mac
        train_result['recall_macro'] = re_mac
        train_result['precision_macro'] = pre_mac

        print(train_result)

        # Calculate the average loss over all of the batches.
        avg_train_loss = total_train_loss / len(train_dataloader)
        avg_train_loss = avg_train_loss.item()
        
        # Measure how long the validation run took.
        train_time = format_time(time.time() - t1)
        
        print("  Train loss: {0:.3f}".format(avg_train_loss))
        print("  Train took: {:}".format(train_time))
        
        # ========================================
        #     TEST ON THE VALIDATION DATASET
        # ========================================
        # after the completion of each training epoch, measure our performance on
        # our training set
        
        print("")
        print("Running Validation set...")

        t2 = time.time()
        
        # tracking variables for train set
        total_val_loss = 0
        all_val_preds = []
        all_val_labels_ids = []
        nll_val_loss = torch.nn.CrossEntropyLoss(ignore_index=-1) #loss

        # evaluate data for one epoch
        for batch in val_dataloader:
            
            # unpack this training batch from our dataloader
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)
            
            # tell pytorch not to bother with constructing the compute graph during
            # the forward pass, since this is only needed for backprop (training)
            with torch.no_grad():        
                model_outputs = transformer(b_input_ids, attention_mask=b_input_mask)
                hidden_states = model_outputs[-1]
                _, logits, probs = discriminator(hidden_states)
                ###log_probs = F.log_softmax(probs[:,1:], dim=-1)
                filtered_logits = logits[:,0:-1]
                # accumulate the test loss
                total_val_loss += nll_val_loss(filtered_logits, b_labels)
                
            # accumulate the predictions and the input labels
            _, preds = torch.max(filtered_logits, 1)
            all_val_preds += preds.detach().cpu()
            all_val_labels_ids += b_labels.detach().cpu()

        # report the final accuracy for this validation run
        all_val_preds = torch.stack(all_val_preds).numpy()
        all_val_labels_ids = torch.stack(all_val_labels_ids).numpy()
        
        # covert back to labels
        label_map2 = dict((v,k) for k,v in label_map.items())
        all_val_preds = [label_map2[x] for x in all_val_preds]
        all_val_preds = convert_labels(all_val_preds)
        
        all_val_labels_ids = [label_map2[x] for x in all_val_labels_ids]
        all_val_labels_ids = convert_labels(all_val_labels_ids)

        #val_accuracy = np.sum(all_val_preds == all_val_labels_ids) / len(all_val_preds)
        #print("  Validation Accuracy: {0:.4f}".format(val_accuracy))
        
        # evaluate metrics
        f1_mi = f1_score(y_true=all_val_labels_ids, y_pred=all_val_preds, average='micro')
        re_mi = recall_score(y_true=all_val_labels_ids, y_pred=all_val_preds, average='micro')
        pre_mi = precision_score(y_true=all_val_labels_ids, y_pred=all_val_preds, average='micro')
    
        f1_mac = f1_score(y_true=all_val_labels_ids, y_pred=all_val_preds, average='macro')
        re_mac = recall_score(y_true=all_val_labels_ids, y_pred=all_val_preds, average='macro')
        pre_mac = precision_score(y_true=all_val_labels_ids, y_pred=all_val_preds, average='macro')
    
        val_result = {}
        val_result['f1_micro'] = f1_mi
        val_result['recall_micro'] = re_mi
        val_result['precision_micro'] = pre_mi
    
        val_result['f1_macro'] = f1_mac
        val_result['recall_macro'] = re_mac
        val_result['precision_macro'] = pre_mac

        print(val_result)

        # Calculate the average loss over all of the batches.
        avg_val_loss = total_val_loss / len(val_dataloader)
        avg_val_loss = avg_val_loss.item()
        
        # Measure how long the validation run took.
        val_time = format_time(time.time() - t2)
        
        print("  Validation loss: {0:.3f}".format(avg_val_loss))
        print("  Validation took: {:}".format(val_time))
        
        # Record all statistics from this epoch.
        training_stats.append(
            {
                'Epoch': epoch_i + 1,
                'Training Loss generator': avg_train_loss_g,
                'Training Loss discriminator': avg_train_loss_d,
                'Training Loss': avg_train_loss,
                'Validation Loss': avg_val_loss,
                'Test Loss': avg_test_loss,
                'Training Result': train_result,
                'Validation Result': val_result,
                'Test Result': test_result,
                'Training Time': training_time,
                'Validation Time': val_time,
                'Test Time': test_time
            }
        )
        
        # val_accuracy
        if (val_result['f1_macro'] > best_train_metric):
            best_train_metric = val_result['f1_macro']
            best_epoch = epoch_i
            
            # save model
            final_dataset = labeled_examples
            if (len(unlabeled_examples) != 0):
                final_dataset += unlabeled_examples
            torch.save({'dataset': final_dataset,
                        'class_names': label_list,
                        'pretrained_model': model_name,
                        'history': training_stats,
                        
                        'transformer': transformer,
                        'discriminator': discriminator
                        }, 'model_' + model_name.replace('/','_') + '.bin')  
        

        history_point = {
                'Epoch': epoch_i + 1,
                'Training Loss generator': avg_train_loss_g,
                'Training Loss discriminator': avg_train_loss_d,
                'Training Loss': avg_train_loss,
                'Validation Loss': avg_val_loss,
                'Test Loss': avg_test_loss,
                'Training Result': train_result,
                'Validation Result': val_result,
                'Test Result': test_result,
                'Training Time': training_time,
                'Validation Time': val_time,
                'Test Time': test_time
            }
            
    result_dict = {}
    result_dict['best_train_metric'] = best_train_metric
    result_dict['best_epoch'] = best_epoch
    result_dict['training_stats'] = training_stats
    write_single_dict_to_json_file('history_file_' + model_name.replace('/','_') + '.json', result_dict, file_access = 'w')    

    print("\nTraining complete!")
    print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))

class CategoryClassifier(nn.Module):
    def __init__(self, n_classes, pretrained_model = 'bert-base-cased'):
        super(CategoryClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(pretrained_model)
        self.drop = nn.Dropout(p=0.1)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask, return_dict=False):
        _, pooled_output = self.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=return_dict)
        output = self.drop(pooled_output)
        
        # rewrite here
        return self.out(output)   

def predict_dataset(saved_model_name = 'best_model.bin', test_filename = 'dataset/test.json',
            output_filename = 'dataset/test_pred.txt', output_filename_label = 'dataset/test_pred_label.txt', 
            batch_size = 16):

    seed_val = 42
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed_val)
    
    # If there's a GPU available...
    if torch.cuda.is_available():    
        # Tell PyTorch to use the GPU.    
        device = torch.device("cuda")
        print('There are %d GPU(s) available.' % torch.cuda.device_count())
        print('We will use the GPU:', torch.cuda.get_device_name(0))
    # If not...
    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")
    
    #--------------------------------
    #  Transformer parameters
    #--------------------------------
    max_seq_length = 128
    #batch_size = 2

    #--------------------------------
    #  GAN-BERT specific parameters
    #--------------------------------
    # number of hidden layers in the generator, 
    # each of the size of the output space
    num_hidden_layers_g = 2; 
    # number of hidden layers in the discriminator, 
    # each of the size of the input space
    num_hidden_layers_d = 2; 
    # size of the generator's input noisy vectors
    noise_size = 768
    # dropout to be applied to discriminator's input vectors
    out_dropout_rate = 0.5

    # Replicate labeled data to balance poorly represented datasets, 
    # e.g., less than 1% of labeled material
    apply_balance = False

    #--------------------------------
    #  Optimization parameters
    #--------------------------------
    #learning_rate_discriminator = 5e-7
    #learning_rate_generator = 5e-7
    #epsilon = 2e-7
    #num_train_epochs = 5
    multi_gpu = True
    # Scheduler
    #apply_scheduler = False
    warmup_proportion = 0.1
    # Print
    #print_each_n_step = 100

    #--------------------------------
    #  Adopted Tranformer model
    #--------------------------------
    # Since this version is compatible with Huggingface transformers, you can uncomment
    # (or add) transformer models compatible with GAN

    #model_name = "bert-base-multilingual-uncased"
    #model_name = "bert-base-cased"
    #model_name = "bert-base-uncased"
    #model_name = "roberta-base"
    #model_name = "albert-base-v2"
    #model_name = "xlm-roberta-base"

    #--------------------------------
    #  Retrieve the dataset
    #--------------------------------

   
    label_list = read_list_from_json_file('dataset/label_names.json')
    print('label_list: ', label_list)
   
    label_map = {}
    for (i, label) in enumerate(label_list):
        label_map[label] = i

    # unpack model dict
    '''torch.save({'dataset': labeled_examples + unlabeled_examples,
                        'class_names': label_list,
                        'pretrained_model': model_name,
                        'history': training_stats,
                        'classifier_type': classifier_type,
                        'transformer': transformer.state_dict(),
                        'discriminator': discriminator.state_dict()
                        }, 'best_model.bin')'''
                        
    dataset = {}
    class_names = {}
    pretrained_model = ''
    history = {}

    checkpoint = torch.load(saved_model_name)
    dataset = checkpoint['dataset']
    class_names = checkpoint['class_names']
    pretrained_model = checkpoint['pretrained_model']
    history = checkpoint['history']
    
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
    transformer = checkpoint['transformer']
    #transformer = AutoModel.from_pretrained(pretrained_model)
    #transformer = torch.nn.DataParallel(transformer)
    #transformer.load_state_dict(checkpoint['transformer'])
    
    '''parallel_model.load_state_dict(
        torch.load("my_saved_model_state_dict.pth", map_location=str(device))
    )'''

    # DataParallel has model as an attribute
    #transformer = transformer.model
    transformer = transformer.to(device)
    transformer.eval()
    
    # the config file is required to get the dimension of the vector produced by 
    # the underlying transformer
    config = AutoConfig.from_pretrained(pretrained_model)
    hidden_size = int(config.hidden_size)
    # Define the number and width of hidden layers
    #hidden_levels_g = [hidden_size for i in range(0, num_hidden_layers_g)]
    hidden_levels_d = [hidden_size for i in range(0, num_hidden_layers_d)]
    
    #discriminator = Discriminator(input_size=hidden_size, hidden_sizes=hidden_levels_d, num_labels=len(label_list), dropout_rate=out_dropout_rate)
    #discriminator.load_state_dict(checkpoint['discriminator'])
    discriminator = checkpoint['discriminator']
    discriminator = discriminator.to(device)
    discriminator.eval()
    
    # put everything in the GPU if available
    if torch.cuda.is_available():    
        discriminator.cuda()
        transformer.cuda()
        if multi_gpu: transformer = torch.nn.DataParallel(transformer)

    test_examples = get_qc_examples(test_filename)
    test_label_masks = np.ones(len(test_examples), dtype=bool)
    
    
    test_dataloader = generate_data_loader(tokenizer, max_seq_length, batch_size, test_examples, test_label_masks, 
                    label_map, do_shuffle = False, balance_label_examples = False)
    
    
    print("")
    print("Running Test...")

    t0 = time.time()

    # Tracking variables 
    total_test_accuracy = 0
       
    total_test_loss = 0
    nb_test_steps = 0

    all_preds = []
    all_labels_ids = []

    #loss
    nll_loss = torch.nn.CrossEntropyLoss(ignore_index=-1)
        
    for batch in test_dataloader:
            
        # Unpack this training batch from our dataloader. 
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)
        
            
        # Tell pytorch not to bother with constructing the compute graph during
        # the forward pass, since this is only needed for backprop (training).
        with torch.no_grad():        
            model_outputs = transformer(b_input_ids, attention_mask=b_input_mask)
            hidden_states = model_outputs[-1]
            _, logits, probs = discriminator(hidden_states)
            ###log_probs = F.log_softmax(probs[:,1:], dim=-1)
            filtered_logits = logits[:,0:-1]
            # Accumulate the test loss.
            total_test_loss += nll_loss(filtered_logits, b_labels)
                
        # Accumulate the predictions and the input labels
        _, preds = torch.max(filtered_logits, 1)
        all_preds += preds.detach().cpu()
        all_labels_ids += b_labels.detach().cpu()

    # Report the final accuracy for this validation run.
    all_preds = torch.stack(all_preds).numpy()
    all_labels_ids = torch.stack(all_labels_ids).numpy()
    
    print('all_preds: ', all_preds)
    print('all_labels_ids: ', all_labels_ids)
    
    # convert back to labels
    label_map2 = dict((v,k) for k,v in label_map.items())
    all_preds = [label_map2[x] for x in all_preds]
    all_preds = convert_labels(all_preds)
        
    all_labels_ids = [label_map2[x] for x in all_labels_ids]
    all_labels_ids = convert_labels(all_labels_ids)

    # evaluate metrics
    f1_mi = f1_score(y_true=all_labels_ids, y_pred=all_preds, average='micro')
    re_mi = recall_score(y_true=all_labels_ids, y_pred=all_preds, average='micro')
    pre_mi = precision_score(y_true=all_labels_ids, y_pred=all_preds, average='micro')
    
    f1_mac = f1_score(y_true=all_labels_ids, y_pred=all_preds, average='macro')
    re_mac = recall_score(y_true=all_labels_ids, y_pred=all_preds, average='macro')
    pre_mac = precision_score(y_true=all_labels_ids, y_pred=all_preds, average='macro')
    
    test_result = {}
    test_result['f1_micro'] = f1_mi
    test_result['recall_micro'] = re_mi
    test_result['precision_micro'] = pre_mi
    
    test_result['f1_macro'] = f1_mac
    test_result['recall_macro'] = re_mac
    test_result['precision_macro'] = pre_mac

    print('test_result: ', test_result)
    
    #test_accuracy = np.sum(all_preds == all_labels_ids) / len(all_labels_ids)
    #print(" Test Accuracy: {0:.4f}".format(test_accuracy))
    
    # write file output
    # data_list = []
    # write_list_to_json_file(output_filename, data_list)
    
    # calculate the average loss over all of the batches.
    avg_test_loss = total_test_loss / len(test_dataloader)
    avg_test_loss = avg_test_loss.item()
        
    # measure how long the validation run took.
    test_time = format_time(time.time() - t0)
        
    print("  Test loss: {0:.4f}".format(avg_test_loss))
    print("  Test took: {:}".format(test_time))
    

def main(args):
    if (args.mode == 'train'):
        train_model(model_name = args.model_name, 
            learning_rate_discriminator = float(args.lr_discriminator), learning_rate_generator = float(args.lr_generator), 
            num_train_epochs = args.epochs, batch_size = args.batch_size)
             

    elif (args.mode == 'test'):
        predict_dataset(saved_model_name = args.model_path, 
            test_filename = args.test_file, output_filename = args.out_test_file, 
            output_filename_label = args.out_test_label_file)

#...............................................................................     
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Training Parameter')
    parser.add_argument('--mode', type=str, default='train') # or test
    parser.add_argument('--model_name', type=str, default='bert-base-cased') # or test
    parser.add_argument('--lr_discriminator', type=str, default='5e-5') 
    parser.add_argument('--lr_generator', type=str, default='5e-5')
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--model_path', type=str, default='model_bert-base-cased.bin')
    parser.add_argument('--test_file', type=str, default='dataset/test.json')
    parser.add_argument('--out_test_file', type=str, default='dataset/test_pred.json')
    parser.add_argument('--out_test_label_file', type=str, default='dataset/test_label.json')
  
    args = parser.parse_args()     
    main(args)

# python gan.py --mode "train" --model_name "bert-base-cased" --lr_discriminator 2e-5 --lr_generator 2e-5 --epochs 25 --batch_size 8
# python gan.py --mode "test" --model_path "model_bert-base-cased.bin"  --test_file "dataset/test.json"