from file_io import *
import random

emotion_list = ['anger', 'brain dysfunction (forget)', 'emptiness', 'hopelessness', 'lack of energy', 'loneliness', 'sadness', 'self hate', 'suicide intent', 'worthlessness']


def manual_annot(input_file = 'dataset/agreement/final_dataset_no_label.json', output_file = 'dataset/agreement/final_dataset_no_label1.json'):



    dataset = read_list_from_jsonl_file(input_file)

    print('**************************************************')
    print('**************************************************')
    print('HUMAN EVALUATION')
    print('Instructions: ')
    print('---', '1. Read the following text carefully and classify it into these categories: [anger, brain dysfunction (forget), emptiness, hopelessness, lack of energy, loneliness, sadness, self hate, suicide intent, worthlessness].')


    print('---', '2. For each category, rate from 0-1:')
    print('------', '0: No, the text must not be classified into this category.')
    print('------', '1: Yes, the text must be classified into this category.')
    #print('------', '2: I have no idea.')

    print('**************************************************')
    print('**************************************************')
    
    

    index = 1
    n = len(dataset)
    for item in dataset[0:1]:

        #[anger, brain dysfunction (forget), emptiness, hopelessness, lack of energy, loneliness, sadness, self hate, suicide intent, worthlessness]
        annot_dict = {}
        print('Numer: ', str(index) + '/' + str(n))
        print('Text: ', item['text'])
        print('Classify the text into these categories.')
        print('--------------------------------------')

        anger = input("anger (0-1): ")
        brain_dysfunction = input("brain dysfunction (forget) (0-1): ")
        emptiness = input("emptiness (0-1): ")
        hopelessness = input("hopelessness (0-1): ")
        lack_of_energy = input("lack of energy (0-1): ")
        loneliness = input("loneliness (0-1): ")
        sadness = input("sadness (0-1): ")
        self_hate = input("self hate (0-1): ")
        suicide_intent = input("suicide intent (0-1): ")
        worthlessness = input("worthlessness (0-1): ")

        annot_dict['anger'] = anger
        annot_dict['brain_dysfunction'] = brain_dysfunction
        annot_dict['emptiness'] = emptiness
        annot_dict['hopelessness'] = hopelessness
        annot_dict['lack_of_energy'] = lack_of_energy
        annot_dict['loneliness'] = loneliness
        annot_dict['sadness'] = sadness
        annot_dict['self_hate'] = self_hate
        annot_dict['suicide_intent'] = suicide_intent
        annot_dict['worthlessness'] = worthlessness
        
        item['coder'] = annot_dict
    
        # reset 
        del anger, brain_dysfunction, emptiness, hopelessness, lack_of_energy, loneliness, sadness, self_hate, suicide_intent, worthlessness 
    
        print('*************************')
        print('*************************')

        index += 1

    # save results
    write_list_to_jsonl_file(output_file, dataset, file_access = 'w')

def create_annot_dataset():
    dataset1 = read_list_from_jsonl_file('dataset/final_dataset.json')
    random.shuffle(dataset1)
    dataset1 = dataset1[0:100]
    write_list_to_jsonl_file('dataset/agreement/final_dataset.json', dataset1, file_access = 'w')
    
    for item in dataset1:
        item['emotions'] = []
    write_list_to_jsonl_file('dataset/agreement/final_dataset_no_label.json', dataset1, file_access = 'w')

    dataset2 = read_list_from_jsonl_file('dataset/final_dataset_gpt.json')
    new_dataset2 = []
    for item1 in dataset1:
        for item2 in dataset2:
            if(item1['id'] == item2['id']):
                '''if (len(item2['emotions']) == 0):
                    create_annot_dataset()'''
                new_dataset2.append(item2)
    dataset2 = new_dataset2

    print(len(dataset1), len(dataset2))      
    
    write_list_to_jsonl_file('dataset/agreement/final_dataset_gpt.json', dataset2, file_access = 'w')

if __name__ == "__main__":
    #create_annot_dataset()
    manual_annot()



'''dataset1 = read_list_from_jsonl_file('dataset/final_dataset_1.json')
dataset5 = read_list_from_jsonl_file('dataset/final_dataset_gpt.json')

def search(id_item, dataset):
    for item in dataset:
        if (id_item == item['id']):
            return item
            
    return {}
    
# gpt dataset
for item in dataset5:

    emotions = []
    emotions = item['emotions']
    
    if (type(item['emotions']) is list):
        emotions = [x.lower().replace("'",'').replace("[",'').replace('\"', '').replace(']', '') for x in emotions]
        
    else:
        try:
            emotions = item['emotions']
            emotions = ast.literal_eval(emotions)
            emotions = [x.lower() for x in emotions]
            
        except:
            try:
                emotions = item['emotions'].split(',')
                emotions = [x.lower() for x in emotions]
            except:
                pass
    
    new_emotions = []
    for feel in emotions:
        if (feel in emotion_list):
            new_emotions.append(feel)
    
    emotions = list(set(new_emotions))
    item['emotions'] = new_emotions
    print('emotions: ', new_emotions)
    print('----------------')

# sort gpt dataset
new_dataset5 = []
for item in dataset1:
    
    temp = search(item['id'], dataset5)
    if temp:
        new_dataset5.append(temp)
    else:
        item['emotions'] = []
        new_dataset5.append(item)
dataset5 = new_dataset5[:]

write_list_to_jsonl_file('dataset/final_dataset_gpt.json', dataset5, file_access = 'w')'''