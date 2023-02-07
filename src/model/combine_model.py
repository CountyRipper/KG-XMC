'''
This model is used to combine or matching . It provides least 4 types: cross-encoder,
'''
import os
import pickle
import torch
from typing import List
from sentence_transformers import SentenceTransformer, util, CrossEncoder
from tqdm import tqdm
from src.utils.premethod import load_map, read_label_text
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def combine(pred_dir,reference_dir,model_name,data_dir,output_dir=None)-> List[List[str]]:
    if not os.path.exists(pred_dir):
        print(f'{pred_dir} not exists, return. ')
        return None
    model_b = SentenceTransformer(model_name_or_path=model_name,device=device)
    #model_b = SentenceTransformer('all-MiniLM-L12-v2',device=device)
    print(f'combine model_name: {model_name}')
    print('pred_data: '+pred_dir)
    print('reference:'+reference_dir)
    print('write into: '+output_dir)
    pred_list=read_label_text(pred_dir)
    all_label_list=load_map(reference_dir)

    if not os.path.exists(data_dir+"all_labels.pkl"):
        embeddings_all = model_b.encode(all_label_list,convert_to_tensor=True,device=device)
        with open(data_dir+"all_labels.pkl", "wb") as fOut:
            pickle.dump({'embeddings': embeddings_all}, fOut, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open(data_dir+'all_labels.pkl', "rb") as fIn:
            stored_data = pickle.load(fIn)
            #stored_sentences = stored_data['sentences']
            embeddings_all = stored_data['embeddings']
    for i in tqdm(range(len(pred_list))):
        no_equal_list=[]
        for ind,each_label in enumerate(pred_list[i]):
            #此处可以考虑不换位置，但是会更复杂，不确定是否会影响结果，单纯extend的话有可能劣化结果
            if each_label not in all_label_list:
                no_equal_list.append({'ind':ind,'label':each_label})
            #对于每一个不在已存在标签列表中的label，计算得到最相似的标签
        if len(no_equal_list)==0:
            continue
        t_list = list(map(lambda x: x['label'], no_equal_list))
        embeddings_pre = model_b.encode(t_list, convert_to_tensor=True,device=device)
        cosine_score = util.cos_sim(embeddings_pre,embeddings_all)
        #cosine_score是一个len(no_equal_list)行，(all_label_list)列的一个矩阵
        #cosine_score的长度一定等于no_equal_list
        flag = torch.zeros(len(all_label_list),device=device)
        for j in range(len(cosine_score)):
            this_score = torch.add(cosine_score[j],flag)
            max_ind = torch.argmax(this_score)
            #while all_label_list[max_ind] in pred_list: #if prelist has this candidate label
            #    cosine_score[j][max_ind]=0
            #    max_ind = cosine_score[j].argmax(0)
            no_equal_list[j]['label'] =  all_label_list[max_ind]
            flag[max_ind] = -2.0
        for j in no_equal_list:
            pred_list[i][j['ind']] = j['label']
    if output_dir:
        print('write into: '+output_dir)
        with open(output_dir,'w+')as w1:
            for row in pred_list:
                w1.write(",".join(row)+'\n')