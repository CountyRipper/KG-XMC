import pickle
from typing import List
from sentence_transformers import SentenceTransformer, util
import torch
from src.utils.decorators import func_log
from tqdm import tqdm
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
    
@func_log
def p_at_k(dir, src_label_dir,pred_label_dir,outputdir=None)->list:
    #src_label_dir = dir+src_label_dir
    #pred_label_dir = os.path.join(dir,'res',pred_label_dir)
    print("p_at_k:"+'\n')
    print("src_label: "+src_label_dir)
    print("pred_label: "+pred_label_dir)
    p_at_1_count=0
    p_at_3_count = 0
    p_at_5_count = 0
    src_label_list=[]
    pred_label_list=[]
    src_label_list = read_label_text(src_label_dir)
    pred_label_list = read_label_text(pred_label_dir)
    num1=len(src_label_list)
    num2 = len(pred_label_list)
    if num1!=num2:
        print("num error")
        return 
    else:
        #recall_100 = get_recall_100(src_label_list,pred_label_list)
        for i in range(num1):
            p1=0 
            p3=0
            p5=0
            for j in range(len(pred_label_list[i])):
                if pred_label_list[i][j] in src_label_list[i]:
                    if j<1:
                        p1+=1
                        p3+=1
                        p5+=1
                    if j>=1 and j <3:
                        p3+=1
                        p5+=1
                    if j>=3 and j<5:
                        p5+=1
            p_at_1_count+=p1
            p_at_3_count+=p3
            p_at_5_count+=p5
        p1 = p_at_1_count/len(pred_label_list)
        p3 = p_at_3_count/ (3*len(pred_label_list))
        p5 = p_at_5_count/ (5*len(pred_label_list))
        print('p@1= '+str(p1))
        print('p@3= '+str(p3))
        print('p@5= '+str(p5))
        #print(f'recall@100 = {recall_100:>4f}')
        if outputdir:
            with open(outputdir,'a+')as w:
                w.write("\n")
                #now_time = datetime.datetime.now()
                #time_str = now_time.strftime('%Y-%m-%d %H:%M:%S')
                #w.write("time: "+time_str+"\n")
                w.write("src_label: "+src_label_dir+"\n")
                w.write('pred_label: '+ pred_label_dir+"\n")
                w.write("p@1="+str(p1)+"\n")
                w.write("p@3="+str(p3)+"\n")
                w.write("p@5="+str(p5)+"\n")
                #w.write(f"recall@100={recall_100:>4f}")
        return [p1,p3,p5]
    
def construct_rank_train(data_dir,model_name,label_map_dir,ground_index_dir,src_text_dir,output_index=None,output_label=None):
    '''
    调用untrained simces或者sentence-transformer排序所有的labels，然后选取前10/5个语义高度相关但是negetive的label作为负标签    
    '''
    print(f'label_map: {label_map_dir}')
    print(f'ground_index_dir: {ground_index_dir}')
    print(f'src_text_dir: {src_text_dir}')
    label_map = load_map(label_map_dir)
    ground_index = read_index(ground_index_dir)
    src_text = read_text(src_text_dir)
    with open(data_dir+'all_labels.pkl', "rb") as fIn:
        stored_data = pickle.load(fIn)
        #stored_sentences = stored_data['sentences']
        embeddings_all = stored_data['embeddings']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SentenceTransformer(model_name_or_path=model_name,device=device)
    #label_text_list = transfer_indexs_to_labels(label_map,ground_index)
    embeddings_src = model.encode(src_text, convert_to_tensor=True,device=device)
    cos_scores = util.cos_sim(embeddings_src,embeddings_all)
    # scores = []
    # for i in cos_scores:
    #     scores.append([[ind, e] for ind, e in enumerate(i)])
    # matrix each line is the socre of all labels for single text record
    un_contain_list = []
    res = []
    for i in tqdm(range(len(cos_scores))):
        count = 7 
        '''调整为5可以减少negative num数量'''
        tmp = []
        flag = torch.zeros(len(embeddings_all),device=device)
        while count>0:
            this_score = torch.add(cos_scores[i],flag)
            max_ind = torch.argmin(this_score).item()
            if  max_ind not in ground_index[i]:
                '''不在，说明是negative'''
                tmp.append(max_ind)
                count-=1
            '''是ground，直接标记以下就可以了'''
            flag[max_ind] = 2.0
        un_contain_list.append(tmp)
    for i in range(len(ground_index)):
        ground_index[i].extend(un_contain_list[i])
    if output_index:
        with open(output_index,'w') as w:
            for row in ground_index:
                w.write(",".join(map(lambda x: str(x),row))+"\n")
    return res