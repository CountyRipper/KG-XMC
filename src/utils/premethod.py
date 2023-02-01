from typing import List
from decorators import func_log
@func_log
def read_text(src)->List:
    res = []
    with open(src,'r') as r:
        for i in r:
            res.append(i.strip())
    return res

@func_log
def read_index(src)->List:
    res=[]
    with open(src,'r') as r:
        for i in r:
            res.append(i.strip().split(","))
@func_log
def load_map(src)->List:
    '''
    load label index map 
    src = ./dataset/data-name/output-items.txt
    return  label List, which has label index information.
    '''
    label_map = []
    with open(src, 'r') as r:
        for i in r:
            label_map.append(i.strip())
    return label_map
                    
    