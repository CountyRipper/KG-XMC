'''
this is the start script of whole system, including keyphrase generation, matching, and ranking
'''

from argparse import ArgumentParser
import os

import torch
from src.model.hg_keyphrase import modeltrainer
from src.model.rank_model import label_rank, rank_train
from src.model.combine_model import combine
from src.model.keyphrase_model import KG_Model, kg_predict, kg_train
from src.utils.premethod import p_at_k


def run(args:ArgumentParser):
    model_time1=None
    model_time2=None
    model_time3=None
    model_time4=None
    model_time5=None
    model_time6=None
    model_time7=None
    #datadir = ['./dataset/wiki10-31k']
    # part 1 : check keyphrase args
    
    torch.cuda.empty_cache()
    if args.datadir == './dataset/amazoncat-13k/':
        trainer = modeltrainer(args)
        if args.is_kg_train:        
            trainer.train()
        if args.is_kg_pred:
            if not os.path.exists(os.path.join(args.datadir,'res')):
                os.mkdir(os.path.join(args.datadir,'res'))
            if args.is_kg_pred_trn:
                trainer.predicting(args.kg_savedir,src_dataname=os.path.join(args.datadir,'X.trn.txt'),
                                   output_dir=os.path.join(args.datadir,'res','trn_pred_'+args.kg_type+'.txt'))
            if args.is_kg_pred_tst:
                trainer.predicting(args.kg_savedir,src_dataname=os.path.join(args.datadir,'X.tst.txt'),
                                   output_dir=os.path.join(args.datadir,'res','tst_pred_'+args.kg_type+'.txt'))
    else: 
        if args.is_kg_train:
            kg_train(args)  
        model = KG_Model(args).load_from_checkpoint(os.path.join(args.datadir,args.kg_savedir))  
        if args.is_kg_pred:
            if not os.path.exists(os.path.join(args.datadir,'res')):
                os.mkdir(os.path.join(args.datadir,'res'))
            if args.is_kg_pred_trn:
                kg_predict(model,src_dir=os.path.join(args.datadir,'X.trn.txt'),
                       output_dir=os.path.join(args.datadir,'res','trn_pred_'+args.kg_type+'.txt'),data_size=8)
            if args.is_kg_pred_tst:
                kg_predict(model,src_dir=os.path.join(args.datadir,'X.tst.txt'),
                       output_dir=os.path.join(args.datadir,'res','tst_pred_'+args.kg_type+'.txt'),data_size=8)
    # part 2: check combine args, combine
    if args.is_combine:
        combine(pred_dir=os.path.join(args.datadir,'res','trn_pred_'+args.kg_type+'.txt'),
                reference_dir=os.path.join(args.datadir,'output-items.txt'),model_name=args.combine_model_name,
                data_dir=args.datadir,output_dir=os.path.join(args.datadir,'res','trn_combine_'+args.kg_type+args.combine_type+'.txt'))
        combine(pred_dir=os.path.join(args.datadir,'res','tst_pred_'+args.kg_type+'.txt'),
                reference_dir=os.path.join(args.datadir,'output-items.txt'),model_name=args.combine_model_name,
                data_dir=args.datadir,output_dir=os.path.join(args.datadir,'res','tst_combine_'+args.kg_type+args.combine_type+'.txt'))
    #part 3: check rank args, rank train, rank
    rank_model_save_dir = os.path.join(args.datadir,args.rankmodel_save)
    if args.is_rank_train:
        pred_sdir=os.path.join(args.datadir,'res','trn_combine_'+args.kg_type+args.combine_type+'.txt')
        #pred_dir=os.path.join(args.datadir,'Y.trn.txt')
        rank_train(data_dir=args.datadir,model_name=args.rank_model,
                   src_text=os.path.join(args.datadir,'X.trn.txt'),pred_dir=pred_sdir,
                   model_save_dir=rank_model_save_dir,batch_size=args.rank_batch,epochs=args.rank_epoch)
    src_labels = os.path.join(args.datadir,'res','tst_combine_'+args.kg_type+args.combine_type+'.txt')
    output_text = os.path.join(args.datadir,'res','tst_rank_'+args.kg_type+args.combine_type+args.rank_type+'.txt')
    output_index = os.path.join(args.datadir,'res','tst_rank_index_'+args.kg_type+args.combine_type+args.rank_type+'.txt')
    if args.is_rank:
        
        label_rank(src_labels=src_labels,src_text=os.path.join(args.datadir,'X.tst.txt'),
                   model_type=args.rank_type,model_name=rank_model_save_dir,label_map=os.path.join(args.datadir,'output-items.txt'),
                   output_text=output_text,output_index=output_index)
    res_output_dir = os.path.join(args.datadir,'res_'+args.kg_type+args.combine_type+args.rank_type+'.txt')
    print(args)
    p_at_k(args.datadir,
           src_label_dir=os.path.join(args.datadir,'Y.tst.txt'),
           pred_label_dir=output_index,outputdir=res_output_dir)
    