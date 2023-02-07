'''
This model is for ranking candidates, inclding training and predicting.
'''
import datetime
import math
import re
import time
from typing import List
from sentence_transformers import SentenceTransformer, InputExample, losses,CrossEncoder,util
from torch.utils.data import DataLoader
import os
import torch
from tqdm import tqdm
from src.utils.premethod import load_map, read_index, read_label_text, read_text, transfer_indexs_to_labels, transfer_labels_to_index
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def rank_train(data_dir,model_name,src_text,pred_dir,model_save_dir,batch_size,epochs):    
    '''
    src_text: train set的documents
    pred_dir: train pred/combine labels(text), not index
    ground_truth_label: 
    '''
    fine_tune_list = []
    #raw_text_list = []
    label_list=[]
    pred_label_list=[]
    label_score_list=[]
    print("text_data: "+src_text)
    print("train_pred_data: "+pred_dir)
    print("model_save_dir: "+model_save_dir)
      
    train_texts = read_text(os.path.join(data_dir,"X.trn.txt"))
    train_index = read_index(os.path.join(data_dir,"Y.trn.txt"))
    label_map = load_map(os.path.join(data_dir,"output-items.txt"))
    ground_train_labels = transfer_indexs_to_labels(label_map,train_index) #list,List[List[str]]
    pred_label_list = read_label_text(pred_dir)
    #print(str(raw_text_list[0])+'\n'+str(pred_label_list[0])+'\n'+str(label_list[0]))
    train_data_un = [InputExample(texts=[i, i]) for m in pred_label_list for i in m ]
    #fine_tune_unsup.append(InputExample(InputExample(texts=[s, s] for s in label_list)))
    for i in range(len(pred_label_list)):
        for each in pred_label_list[i]:
            label_len_list = len(ground_train_labels[i])
            if each in ground_train_labels[i]:
                #label_score = 0.5+0.5 *(label_len_list - label_list[i].index(each))/label_len_list
                label_score = 1.0
                #fine_tune_unsup.append(InputExample(InputExample(texts=[s, s]) for s in raw_text_list[i].rstrip()))
                fine_tune_list.append(InputExample(texts=[train_texts[i].rstrip(), each], label=label_score))
                label_score_list.append(str(i) + ' ' + each+ ' ' +str(label_score))
            else:
                fine_tune_list.append(InputExample(texts=[train_texts[i].rstrip(), each], label=0.0))
                label_score_list.append(str(i) + ' ' + each+ ' 0')

    with open (data_dir+"build_labels.txt", "w+") as fb:
        for each in label_score_list:
            fb.write(each)
            fb.write("\n")    
    num_epoch = epochs
    if re.match('\w*cross-encoder\w*',model_name,re.I):
        model = CrossEncoder(model_name, num_labels=1)
        train_dataloader = DataLoader(fine_tune_list, shuffle=True, batch_size=batch_size)
        # shuffle=True
        print("batch_size=",batch_size)
        # Configure the training
        warmup_steps = math.ceil(len(train_dataloader) * num_epoch * 0.1) #10% of train data for warm-up
        #logger.info("Warmup-steps: {}".format(warmup_steps))
        model.fit(train_dataloader=train_dataloader,
                  epochs=num_epoch,
                  warmup_steps=warmup_steps,
                  #用curr
                  output_path=model_save_dir)
        '''需要修改,保存check路径'''
        model.save(model_save_dir)
    #bi-encoder
    elif 'sup-simcse' in model_name:
        if 'unsup-simcse'in model_name:
            print("unsupervised")
            model = SentenceTransformer(model_name_or_path=model_name,device=device)   
            train_loss = losses.MultipleNegativesRankingLoss(model) 
            #train_dataloader_un = DataLoader(fine_tune_unsup, shuffle=True, batch_size=batch_size)
            train_dataloader = DataLoader(train_data_un, batch_size=64, shuffle=True)
            warmup_steps = math.ceil(len(train_dataloader) * num_epoch * 0.1) #10% of train data for warm-up
            #   logger.info("Warmup-steps: {}".format(warmup_steps))
            
            model.fit(train_objectives=[(train_dataloader, train_loss)],
                epochs=num_epoch,
                warmup_steps=warmup_steps,
                #train_loss=train_loss,
                #用curr
                output_path=model_save_dir)
            model.save(model_save_dir)
            return
        else:
            model = SentenceTransformer(model_name_or_path=model_name,device=device)
            train_loss = losses.CosineSimilarityLoss(model)         
    else:
        model = SentenceTransformer(model_name,device=device)
        train_loss = losses.CosineSimilarityLoss(model)
        #model = SentenceTransformer('all-MiniLM-L6-v2')
    train_dataloader = DataLoader(fine_tune_list, shuffle=True, batch_size=batch_size)
    #evaluator = evaluation.EmbeddingSimilarityEvaluator()
    shuffle=True
    #print("batch_size="+ "24")
    # Configure the training
    warmup_steps = math.ceil(len(train_dataloader) * num_epoch * 0.1) #10% of train data for warm-up
    #logger.info("Warmup-steps: {}".format(warmup_steps))
    model.fit(train_objectives=[(train_dataloader, train_loss)],
            epochs=num_epoch,
            warmup_steps=warmup_steps,
            #train_loss=train_loss,
            #用curr
            output_path=model_save_dir)
    model.save(model_save_dir)

def label_rank(src_labels,src_text,model_type,model_name,label_map=None,output_text=None,output_index=None)->List[List[int]]:
    '''
    src: combined labels, which need to ranked.
    model_type: for separated model type
    output_text: the ranked result of labels.
    output_index: the ranked results of index.
    '''
    rank_model = None
    if model_type in 'bi':
        rank_model = SentenceTransformer(model_name)
    elif model_type in 'cr':
        rank_model = CrossEncoder(model_name)
    elif model_type in 'sim':
        rank_model = SentenceTransformer(model_name)
    else:
        rank_model = SentenceTransformer(model_name)
    print(f'labels_texts: {src_labels}')
    pred_labels = read_label_text(src_labels)
    text_list=read_text(src_text)
    ranked_list=[]#保存排序好的列表
    scores_list=[]#记录分数列表的列表
    num1 = len(pred_labels)
    num2 = len(text_list)
    if num1!=num2:
        print('src_value error')
        return
    for i in tqdm(range(num1)):
        score_list=[]
        #ranked_list=[]
        src_text = text_list[i]
        cur_label_set = pred_labels[i]
        #获取文本和不同标签的embedding
        text_embedding = rank_model.encode(src_text,convert_to_tensor=True)
        label_embedding = rank_model.encode(cur_label_set,convert_to_tensor=True)
        cosine_scores = util.cos_sim(text_embedding, label_embedding)
        #获取之后计算得分
        #cosine_scores应该是一个数组，对应每个标签的优先级
        for ind,each_score in enumerate(cosine_scores[0].tolist()):
            score_list.append([cur_label_set[ind],each_score])
        if i%1000==0:
            print(score_list)
        score_list.sort(key= lambda x:x[1],reverse=True) #按照分数排序
        if i%1000==0:
            print(score_list)
        scores_list.append(score_list)
        ranked_list.append(list(map(lambda x:x[0],score_list)))#抽取label部分
    if output_text:
        with open(output_text,'w+') as w1:
            for row in ranked_list:
                w1.write(",".join(row)+'\n')
        with open(output_text.rstrip(".txt")+"_score.txt",'w+') as w2:
            for row in scores_list:
                w2.write(str(row)+'\n')
    if label_map:
        label_map = load_map(label_map)
        ranked_list = transfer_labels_to_index(label_map,ranked_list)
        if output_index:
            with open(output_index,'w+') as w1:
                for row in ranked_list:
                    w1.write(",".join(map(lambda x: str(x),row))+"\n")                      
    return ranked_list
    
        