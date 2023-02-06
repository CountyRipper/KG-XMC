from typing import List
from src.utils.decorators import func_log
@func_log
def read_text(src)->List:
    res = []
    with open(src,'r') as r:
        for i in r:
            res.append(i.strip())
    return res

@func_log
def read_index(src)->List[List[int]]:
    res=[]
    with open(src,'r') as r:
        for i in r:
            cur = []
            for j in i.strip().split(","):
                cur.append(int(j))
            res.append(cur)
    return res

@func_log
def read_label_text(src)->List[List[str]]:
    res = []
    with open(src,'r') as r:
        for i in r:
            res.append(i.strip().split(","))
    return res

@func_log
def load_map(src)->List[str]:
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

#@func_log
def transfer_indexs_to_labels(label_map,index_lists)->List[List[str]]:
    label_texts = []
    for i in index_lists:
        cur_labels=[] #对于一条记录的labels 做映射
        for j in i:
            cur_labels.append(label_map[j])
        label_texts.append(cur_labels)
    return label_texts

def transfer_labels_to_index(label_map:List[str],label_texts)->List[List[str]]:
    index_list = []
    for i in label_texts:
        cur_indexs = []
        for j in i:
            cur_indexs.append(label_map.index(j))
        index_list.append(cur_indexs)
    return index_list                            
    