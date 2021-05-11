'''
Amazon Review Dataset Transformation Script
author: sjoon-oh @ Github
source: -
'''

keys_2_rmv = [
    # meta file
    'description', 
    'image', 
    'tech1', 
    'tech2', 
    'feature', 
    'rank', 
    'also_buy', 
    'also_view',
    'fit',
    'title',
    'similar_item',
    'main_cat',
    'online code',
    'domestic shipping',
    'international shipping',
    'shipping advisory',
    'isbn-13',
    'isbn-10',
    'shipping',
    '',
    'bookmark',
    'date first listed on amazon',
    'note',
    'listening length',
    'audible.com release date',
    'package dimensions',
    '5.25\" disk',
    '3.5\" disk',
    '3.5\" and 5.25\" disks',
    'product dimensions',
    
    # Review file
    'summary',
    'reviewername',
    'reviewtext',
    # 'unixreviewtime',
    'reviewtime',
    'format',
]

allowed = [
    # None for now
]