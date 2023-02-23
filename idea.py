'''
1.调用原生biencoder 排序所有的ground labels
'''

import os
from src.model.rank_model import label_rank
from src.utils.premethod import *

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
construct_rank_train(data_dir=datadir,model_name='all-MiniLM-L12-v2',
                     label_map_dir=os.path.join(datadir,'output-items.txt'),
                     ground_index_dir=os.path.join(datadir,'Y.trn.txt'),
                     src_text_dir=os.path.join(datadir,'X.trn.txt'),output_index=os.path.join(datadir,'Y_com.trn.txt'))