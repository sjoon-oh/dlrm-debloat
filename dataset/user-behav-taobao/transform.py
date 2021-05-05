'''
Taobao(Alibaba) User Behavior dataset \
Transformation Script
author: sjoon-oh @ Github
source: -
'''

import pandas as pd
import numpy as np

# Load files.
file_path = './UserBehavior.csv'
# file_path = './UserBehaviorTest.csv'

file_path_p1 = './UserBehavior-p1.csv'
file_path_fin = './train-taobao.txt'

# import os.path
# if not os.path.isfile(file_path_fin):

df = pd.read_csv(
        file_path, 
        header=None, 
        index_col=None
    )

df.columns = ['user_id', 'item_id', 'cat_id', 'behavior', 'timestamp']
df['dummy'] = 0

df = df[['behavior', 'dummy', 'user_id', 'item_id', 'cat_id', 'timestamp']] # Reorder

# pv	Page view of an item's detail page, equivalent to an item click
# buy	Purchase an item
# cart	Add an item to shopping cart
# fav	Favor an item

#
# Hashing function: String to Hash
import hashlib

def get_hash_value(in_str):

    blake  = hashlib.blake2b(in_str.encode('utf-8'), digest_size=4)
    return hex(int(blake.hexdigest(), base=16))


# Transformation Phase 1.
# Changing the time format.
import time

def extract_year(t_stamp):
    try: time_format = time.strftime("%Y", time.localtime(t_stamp))
    except Exception: time_format = 0

    return time_format

def extract_month(t_stamp):
    try: time_format = time.strftime("%m", time.localtime(t_stamp))
    except Exception: time_format = 0

    return time_format

def extract_time(t_stamp):
    try: time_format = time.strftime("%H:%M:%S", time.localtime(t_stamp))
    except Exception: time_format = 0

    return time_format

#
# Extraction
df['year'] = df['timestamp'].apply(lambda x: extract_year(x))
df['month'] = df['timestamp'].apply(lambda x: extract_month(x))
df['time'] = df['timestamp'].apply(lambda x: extract_time(x))

df.apply(pd.to_numeric, errors='coerce').fillna(0)

#
# Do categorizing
def categorize_month(m_info):
    ret_string = '0'
    q_tb = [
            list(range(1, 4)),
            list(range(4, 7)),
            list(range(7, 10)),
            list(range(10, 13)),
        ]

    for idx in range(len(q_tb)):
        if int(m_info) in q_tb[idx]: 
            ret_string = idx + 1
            break
    
    return f"{ret_string}Q"


def categorize_time(t_info):
    ret_string = 'unknown'
    s_tb = ['midnight', 'morning', 'afternoon', 'evening', 'night', 'midnight']
    t_tb = [
            list(range(0, 5)),
            list(range(5, 12)),
            list(range(12, 16)),
            list(range(16, 20)),
            list(range(20, 22)),
            list(range(22, 23)),
        ]
    
    for idx in range(len(s_tb)):
        if int(t_info[0:2]) in t_tb[idx]: 
            ret_string = s_tb[idx]
            break

    return f"{ret_string}"

# 
df['quarter'] = df['month'].apply(lambda x: categorize_month(x))
df['time'] = df['time'].apply(lambda x: categorize_time(x))

#
# Delete the original value
df = df.drop(['timestamp'], axis=1)

#
# Save modified.
df.to_csv(file_path_p1, index=False)

#
# Conversion
# 1. Behavior
behavior_type = ['pv', 'buy', 'cart', 'fav']
df['behavior'] = df['behavior'].apply(lambda x: behavior_type.index(x))

# 2. Else
for col in df.columns:
    if col == 'behavior': continue
    if col == 'dummy': continue

    df[col] = df[col].apply(lambda x: str(get_hash_value(str(x))).replace('0x', ''))


print(df.info())
print(df.head(20))
df.to_csv(file_path_fin, sep ='\t', header=False, index=False)

