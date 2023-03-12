'''
1.调用原生biencoder 排序所有的ground labels
'''

import os
from src.model.rank_model import label_rank
from src.utils.premethod import *
from random import shuffle
# ground_label_index = read_index('./dataset/wiki10-31k/Y.trn.txt')
# label_map = load_map('./dataset/wiki10-31k/output-items.txt')
# ground_label=transfer_indexs_to_labels(label_map,ground_label_index)
# with open('./dataset/wiki10-31k/Y_raw.trn.txt','w') as w:
#     for i in ground_label:
#         w.write(",".join(i))
#         w.write('\n')
# label_rank(src_labels='./dataset/wiki10-31k/Y_raw.tst.txt',src_text='./dataset/wiki10-31k/X.tst.txt',
#            model_type='bi',model_name='sentence-transformers/all-MiniLM-L12-v2',label_map='./dataset/wiki10-31k/output-items.txt',
#            output_text=None,output_index='./dataset/wiki10-31k/Y_rank.tst.txt')
# label_rank(src_labels='./dataset/wiki10-31k/Y_raw.trn.txt',src_text='./dataset/wiki10-31k/X.trn.txt',
#            model_type='bi',model_name='sentence-transformers/all-MiniLM-L12-v2',label_map='./dataset/wiki10-31k/output-items.txt',
#            output_text=None,output_index='./dataset/wiki10-31k/Y_rank.trn.txt')
datadir = './dataset/wiki10-31k/'
# construct_rank_train(data_dir=datadir,model_name='all-MiniLM-L12-v2',
#                      label_map_dir=os.path.join(datadir,'output-items.txt'),
#                      ground_index_dir=os.path.join(datadir,'Y.trn.txt'),
#                      src_text_dir=os.path.join(datadir,'X.trn.txt'),output_index=os.path.join(datadir,'Y_com.trn.txt'))
datadir = './dataset/wiki10-31k/'
def sort_shuffle(datadir):
    train_index = read_index(os.path.join(datadir,"Y.trn.txt"))
    test_index = read_index(os.path.join(datadir,"Y.tst.txt"))
    for i in train_index:
        shuffle(i)
    for i in test_index:
        shuffle(i)
    with open(datadir+"Y_shuffle.trn.txt",'w+') as w1:
        for i in train_index:
            w1.write(",".join(list(map(lambda x: str(x),i))))
            w1.write('\n')
    with open(datadir+"Y_shuffle.tst.txt",'w+') as w1:
        for i in test_index:
            w1.write(",".join(list(map(lambda x: str(x),i))))
            w1.write('\n')
                
def sort_by_frequency(datadir):
    train_texts = read_text(os.path.join(datadir,"X.trn.txt"))
    test_texts = read_text(os.path.join(datadir,"X.tst.txt"))
    train_index = read_index(os.path.join(datadir,"Y.trn.txt"))
    test_index = read_index(os.path.join(datadir,"Y.tst.txt"))
    #train_index = list(map(lambda x: x if len(x)<10 else x[0:10],train_index))
    label_map = load_map(os.path.join(datadir,"output-items.txt"))
    train_labels_list = transfer_indexs_to_labels(label_map,train_index) #list,需要转化成text
    index_freq_list = []
    train_texts.extend(test_texts)
    for i in tqdm(range(len(label_map))):
        count = 0
        for j in train_texts:
            if label_map[i] in j:
                count+=1
        index_freq_list.append(count)
    com_list = list(zip(label_map,index_freq_list))
    for i in range(len(train_index)):
        list.sort(train_index[i],key= lambda x:com_list[x][1])
    for i in range(len(test_index)):
        list.sort(test_index[i],key= lambda x:com_list[x][1])

    with open(datadir+"Y_freq.trn.txt",'w+') as w1:
        for i in train_index:
            w1.write(",".join(list(map(lambda x: str(x),i))))
            w1.write('\n')
    with open(datadir+"Y_freq.tst.txt",'w+') as w2:
        for i in test_index:
            w2.write(",".join(list(map(lambda x: str(x),i))))
            w2.write('\n')

sort_shuffle(datadir)