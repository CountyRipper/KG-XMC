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