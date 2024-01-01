import csv
import sys
import time as custom_time
import json
from numpyencoder import NumpyEncoder

maxInt = sys.maxsize
while True: # decrease the maxInt value by factor 10 as long as the OverflowError occurs
    try:
        csv.field_size_limit(maxInt)
        break
    except OverflowError:
        maxInt = int(maxInt/10)


def write_data_to_csv_file(file_name, header_list, data_dict):

    try:
        file = open(file_name, 'a', newline = '', encoding = 'utf-8')
        with file:
            writer = csv.DictWriter(file, fieldnames = header_list)
            #writer.writeheader()
            writer.writerow(data_dict)
    except Exception as e:
        print('Error -- write_data_to_csv_file: ', e)
        pass

        
def write_to_text_file(file_name, data):
    """
        write a text in a new line to file
            file_name: string - file name
            data: string
    """

    try:
        with open(file_name, 'a', encoding='utf-8') as f:
            f.write(data + '\n')
        f.close()
    except Exception as e:
        print('Error -- write_to_text_file: ', e)
    
	
def write_to_new_text_file(file_name, data):
    """
        write text into a new file
            file_name: string - file name
            data: string
    """
    try:
        with open(file_name, 'w', encoding='utf-8') as f:
            if (data == ''):
                f.write(data)
            else:
                f.write(data + '\n')
        f.close()
    except Exception as e:
        print('Error -- write_to_new_text_file: ', e)


def write_list_to_json_file(file_name, data_list, file_access = 'a'):
    """
        write to json file, append new list
            data_list: list - list of data
    """

    try:
        with open(file_name, file_access, encoding='utf-8') as outfile:
            json.dump(data_list, outfile, indent=4, separators=(', ', ': '), ensure_ascii=False, cls=NumpyEncoder)
        outfile.close()
    except Exception as e:
        print('Error -- write_list_to_json_file: ', e)

def write_list_to_jsonl_file(file_name, data_list, file_access = 'a'):
    """
        write to json file, append new list
            data_list: list - list of data
    """

    try:
        with open(file_name, file_access, encoding='utf-8') as outfile:
            for item in data_list:
                #print(item)
                json.dump(item, outfile, separators=(', ', ': '), ensure_ascii=False, cls=NumpyEncoder)
                outfile.write('\n')
        outfile.close()
    except Exception as e:
        print('Error -- write_list_to_jsonl_file: ', e)

def write_list_to_tsv_file(file_name, data_list, delimiter = '\t', file_access = 'a', quoting = csv.QUOTE_NONE):
    """
        write to json file, append new list
            data_list: list - list of data
    """

    try:
        with open(file_name, file_access, encoding='utf-8', newline='') as outfile:
            for item in data_list:
                data_item = [v for k, v in item.items()] # convert to value list
                tsv_output = csv.writer(outfile, delimiter=delimiter, lineterminator='\n', quoting=quoting)
                tsv_output.writerow(data_item)
        
        outfile.close()

    except Exception as e:
        print('Error -- write_list_to_tsv_file: ', e)


def write_single_dict_to_jsonl_file(out_file_name, data_dict, try_no = 0, file_access = 'a', format_json = False):
    """
        write list to *.json file
            out_file_name: string - file name
            data_dict: dict - a single data dictionary
            try_no:  int - the number of times to try to write
            return: void
    """

    try:        
        data_string = json.dumps(data_dict, separators=(', ', ': '), ensure_ascii=False, cls=NumpyEncoder)
        with open(out_file_name, file_access, encoding='utf-8') as outfile:
            if (format_json == False):
                outfile.write(data_string + '\n')
            else:
                outfile.write(data_string)
        outfile.close()
        return True
        
    except Exception as e:
        print('Error -- write_single_dict_to_json_file: ', e)
        try_no += 1
        if (try_no <= 10):
            return write_single_dict_to_jsonl_file(out_file_name, data_dict, try_no, file_access) # try to read file again

    return False
    
def write_single_dict_to_json_file(out_file_name, data_dict, try_no = 0, file_access = 'a', format_json = False):
    """
        write list to *.json file
            out_file_name: string - file name
            data_dict: dict - a single data dictionary
            try_no:  int - the number of times to try to write
            return: void
    """

    try:        
        # indent=4,
        data_string = json.dumps(data_dict, separators=(', ', ': '), ensure_ascii=False, cls=NumpyEncoder)
        with open(out_file_name, file_access, encoding='utf-8') as outfile:
            if (format_json == False):
                outfile.write(data_string + ',\n')
            else:
                outfile.write(data_string)
        outfile.close()
        return True
        
    except Exception as e:
        print('Error -- write_single_dict_to_json_file: ', e)
        try_no += 1
        if (try_no <= 10):
            return write_single_dict_to_json_file(out_file_name, data_dict, try_no, file_access) # try to read file again

    return False

def read_list_from_json_file(out_file_name, format_json = True, try_no = 0):
    """
        load list from *.json file
            out_file_name: string - file name
            return: list - a return list
    """

    result_list = []
    try:
        with open(out_file_name, 'r', encoding='utf-8') as outfile:
            text = outfile.read()
            text = text.strip(',\n') # remove the start & last syntaxes

            if (format_json == False):
                result_list = json.loads('[' + text + ']') 
            else:
                result_list = json.loads(text)
        outfile.close()

    except Exception as e:
        print('Error -- read_list_from_json_file: ', e)
        try_no += 1
        if (try_no <= 10):
            custom_time.sleep(2)
            return read_list_from_json_file(out_file_name, format_json, try_no) # try to read file again
        pass
    
    return result_list
    

def read_list_from_jsonl_file(out_file_name, try_no = 0):
    """
        load list from *.json file
            out_file_name: string - file name
            return: list - a return list
    """

    result_list = []
    i = 0
    try:
        with open(out_file_name, 'r', encoding='utf-8') as outfile:
            for line in outfile:
                item = json.loads(line) 
                result_list.append(item)
                i += 1

        outfile.close()
    except Exception as e:
        print('Error -- read_list_from_jsonl_file: ', e, '-- check line: ', i)
        try_no += 1
        if (try_no <= 10):
            custom_time.sleep(2)
            return read_list_from_jsonl_file(out_file_name, try_no) # try to read file again
        pass
    
    return result_list


def write_list_to_text_file(file_name, data_list, file_access = 'a'): 
    """
        write a text in a new line to file
            file_name: string - file name
            data: string
    """

    try:
        data = ''
        for item in data_list:
            data += item + '\n'
        with open(file_name, file_access, encoding='utf-8') as f:
            f.write(data)
        f.close()
    except Exception as e:
        print('Error -- write_list_from_text_file: ', e)


def read_list_from_text_file(file_name):
    """
        read from text file
            file_name: string - file name
    """
    
    page_list = []
    try:
        with open(file_name, 'r', encoding='utf-8') as f:
            for line in f:
              page_list.append(line.strip())
              #print(line)
        f.close()
    except:
        print('Error -- read_list_from_text_file: ', e)
        with open(file_name, 'a', encoding='utf-8') as f: # create a new empty file
            f.close()

    if (len(page_list) == 1):
        return page_list[0] # return the first element in list
    
    return page_list


def read_from_text_file(file_name):
    """
        write a text in a new line to file
            file_name: string - file name
            data: string
    """
    
    data = ''
    try:
        with open(file_name, 'r', encoding='utf-8') as f:
            data = f.read()
        f.close()
    except:
        pass

    return data

def read_list_from_csv_file(file_name, delimiter = ',', encoding = 'utf-8'):
    """
        write a text in a new line to file
            file_name: string - file name
            return: list
    """
    
    data_list = []
       
    try:
        with open(file_name, 'r', encoding=encoding) as f:
            file = csv.reader(f, delimiter = delimiter)
            for line in file:
                data_list.append(line)
        f.close()
    except Exception as e:
        print('Error -- read_list_from_tsv_file: ', e)
        with open(file_name, 'a', encoding=encoding) as f: # create a new empty file
            f.close()

    if (len(data_list) == 1):
        return data_list[0] # return the first element in list
    
    return data_list
