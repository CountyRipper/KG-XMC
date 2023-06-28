from argparse import ArgumentParser
import os

import torch
from src.model.hg_keyphrase import modeltrainer
from src.model.rank_model import label_rank, rank_train
from src.model.combine_model import combine
from src.model.keyphrase_model import KG_Model, kg_predict, kg_train
from src.utils.premethod import p_at_k
from idea import random_sampling
datadir = './dataset/wiki10-31k'
rank_model = "all-mpnet-base-v2"
kg_type='bart'
combine_type='bi'
random_sampling(datadir=datadir,src_dir='res/tst_combine_'+kg_type+combine_type+'.txt',sampling_count=5)
#parameters
#pred_sdir=os.path.join(datadir,'res','trn_combine_'+kg_type+combine_type+'.txt') #
pred_sdir=os.path.join(datadir,'Y_random_sample.trn.txt')
rank_model_save_dir = os.path.join(datadir,'bi_rank')
rank_train(data_dir=datadir,model_name=rank_model,
                   src_text=os.path.join(datadir,'X.trn.txt'),pred_dir=pred_sdir,
                   model_save_dir=rank_model_save_dir,batch_size=32,epochs=2)
src_labels = os.path.join(datadir,'res','tst_combine_'+kg_type+combine_type+'.txt')
rank_type = 'mp_random'
output_text = os.path.join(datadir,'res','tst_rank_'+kg_type+combine_type+rank_type+'.txt')
output_index = os.path.join(datadir,'res','tst_rank_index_'+kg_type+combine_type+rank_type+'.txt')
label_rank(src_labels=src_labels,src_text=os.path.join(datadir,'X.tst.txt'),
                   model_type=rank_type,model_name=rank_model_save_dir,label_map=os.path.join(datadir,'output-items.txt'),
                   output_text=output_text,output_index=output_index)
res_output_dir = os.path.join(datadir,'res_'+kg_type+combine_type+rank_type+'.txt')

p_at_k(datadir,
           src_label_dir=os.path.join(datadir,'Y.tst.txt'),
           pred_label_dir=output_index,outputdir=res_output_dir)