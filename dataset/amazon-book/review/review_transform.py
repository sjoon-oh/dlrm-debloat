'''
Amazon Review Dataset Transformation Script
author: sjoon-oh @ Github
source: -
'''

import pandas as pd
import numpy as np

import os, json

#
# Fetches dataset.
json_files = [_ for _ in os.listdir() if _[-5:] == '.json']

meta_files = json_files
review_files = [
    meta_files.pop(meta_files.index(_)) \
        for _ in json_files if _[0:4] != 'meta'
    ]

json_files = meta_files + review_files

split_num = 1

import re
from book_config import *

#
# Format the string
def is_html(string):
    keywords = [
        'margin',
        '!important',
        'font-size',
    ]

    _ = False
    for keyword in keywords:
        _ = keyword in string
        if _: break
    
    return _

def re_format(string):
    string = str(string)
    string = re.sub('<.*?>', '', string)
    string = re.sub("\\\\.[^a-zA-Z' ]", '', string) # Remove escape sequence
    string = re.sub('\s+', ' ', string) # remove multiple spaces!

    if is_html(string): string = ''
    else:
        string = string.strip() # Remove spaces both ends
        string = string.replace(':', '') # Unnecessarys
        string = string.replace('&amp;', '')
        string = string.replace('&gt;', '')

        string = string.lower()

    return string

#
# To dense feature
def to_dense(string):
    try:
        if type(string) == str:
            if string[-5:] == 'pages': # Format pages
                string = int(string[:-6])
            elif string[:1] == '$': # Format prices
                string = float(string[1:])
            elif string[-2:] == '.0' : # 'overall' case
                string = int(float(string)) # Make it int. 
            else: pass

    except ValueError:
        pass
    
    return string

#
# To dense feature
def to_dense(string):
    try:
        if type(string) == str:
            if string[-5:] == 'pages': # Format pages
                string = int(string[:-6])
            elif string[:1] == '$': # Format prices
                string = int(float(string[1:])) # Make int!
            else: pass

    except ValueError:
        pass
    
    return string




#
# Distinguish whether the dataset is the review or meta.
def split_dataset(f_name, split_num):
   
    total_lines = 0

    with open(f_name, 'r') as data_file:
        total_lines = len(list(data_file))
    del data_file

    file_size = int(total_lines / split_num)
    file_count = 0

    try:
        with open(f_name, 'r') as data_file: 
            print(f"Conversion to dataframe: {f_name}")

            tf = {}
            export_list = []

            line_count = 0

            for line in data_file:
                line = json.loads(line)
                line_count += 1

                tf = {}

                for key in line:                
                    if re_format(key) not in keys_2_rmv: # process only allowed
                        if type(line[key]) is dict:
                            for subkey in line[key]:
                                new_key = re_format(f"{subkey.strip()}")
                                if new_key not in keys_2_rmv: tf[new_key] = to_dense(line[key][subkey])

                        elif type(line[key]) is list: # Only set the last one.
                            _ = [to_dense(re_format(x)) for x in line[key]]
                            tf[re_format(key)] = _

                        else: tf[re_format(key)] = to_dense(re_format(line[key]))
                
                export_list.append(tf)

                print(f"\r  Processing... {line_count / total_lines * 100: 4.3f}% ({line_count}/{total_lines})", end='')

                if (len(export_list) == file_size or line_count == total_lines):
                    print(f"  Exporting to {f_name.replace('.json', '')}_ps_{file_count}.middle ...")
                    with open(f"{f_name.replace('.json', '')}_ps_{file_count}.middle", "w") as out_file:
                        json.dump(export_list, out_file)

                    del export_list                    
                    export_list = []
                    file_count += 1

        del data_file

    except Exception as e:
        print(e.message, e.args)


import os.path

#
# run script
if __name__ == '__main__':

    print("""Amazon Review Dataset Transformation Script
    \rauthor: sjoon-oh @ Github
    \rsource: -""")

    # json_files = ['temp.json']


    for  _f_name_ in json_files:

        # Check if file exists
        middle_files = [_ for _ in os.listdir() \
            if _[-7:] == '.middle' and \
                _[0:len(_f_name_.replace('.json', ''))] == _f_name_.replace('.json', '')]

        print("Found middle files:")
        for f_name in middle_files: print(f"  {f_name}")

        if len(middle_files) < split_num:
            split_dataset(_f_name_, split_num)
        else: print("Skip splitting.")

        print('Dataframe converting...')
        
        #
        # Step 2: Concat
        middle_files = [_ for _ in os.listdir() \
            if _[-7:] == '.middle' and \
                _[0:len(_f_name_.replace('.json', ''))] == _f_name_.replace('.json', '')]

        #
        # Change to df using processed file

        # df = pd.read_json(middle_files[0])
        # for m_file in middle_files[4:5]:
        #     df = pd.concat([df, pd.read_json(m_file)], axis=0)

        # print('  Done.')

        # print(df.info())
        # print(df.head(200))
        # print(df.isna().sum().sum())







