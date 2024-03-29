import json
import os
from datasets import load_dataset,Dataset
import torch
from tqdm import tqdm
from transformers import (AutoTokenizer,BartTokenizerFast,BartTokenizer, BartForConditionalGeneration,Seq2SeqTrainer, 
                          Seq2SeqTrainingArguments,PegasusForConditionalGeneration, PegasusTokenizerFast,
                          PegasusTokenizer,T5ForConditionalGeneration,T5TokenizerFast,AutoModelForSeq2SeqLM)
import time,datetime
from torch.utils.data import DataLoader

from src.utils.premethod import load_map, read_index, read_text, transfer_indexs_to_labels
class MyData(torch.utils.data.Dataset):
    def __init__(self, encoding, labels):
        self.encoding = encoding
        self.labels= labels
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encoding.items()}
        item['labels'] = torch.tensor(self.labels['input_ids'][idx])
        return item
    def __len__(self):
        return len(self.labels['input_ids'])  # len(self.labels)

class CustomTrainer(Seq2SeqTrainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        # compute custom loss (suppose one has 3 labels with different weights)
        loss_fct = torch.nn.CrossEntropyLoss(weight=torch.tensor([1.0, 2.0, 3.0]))
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

class modeltrainer(object):
    def __init__(self,args) -> None:
        self.datadir = args.datadir
        self.modelname = args.kg_modelname
        self.checkdir = self.datadir +args.kg_checkdir
        self.output = self.datadir + args.kg_savedir
        self.batch_size = args.kg_batch_size
        self.epoch = args.kg_epoch
        self.len_max = args.len_max
        #self.affix = args.affix1
        #self.top_k=args.top_k
        self.learning_rate = args.kg_lr
        self.data_size = args.data_size
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if self.modelname=='facebook/bart-large' or self.modelname=='BART-large' or self.modelname=='Bart-large':
            self.model = BartForConditionalGeneration.from_pretrained("facebook/bart-large",cache_dir='./models').to(self.device)
            self.tokenizer = BartTokenizerFast.from_pretrained(pretrained_model_name_or_path="facebook/bart-large",cache_dir='./models')
        elif self.modelname=='facebook/bart-base' or self.modelname=='BART' or self.modelname=='Bart':
            self.model = BartForConditionalGeneration.from_pretrained("facebook/bart-base",cache_dir='./models').to(self.device)
            self.tokenizer = BartTokenizerFast.from_pretrained(pretrained_model_name_or_path="facebook/bart-base",cache_dir='./models')
        elif self.modelname=='pegasus' or self.modelname=='Pegasus' or self.modelname=='Pegasus-lrage'or self.modelname=='pegasus-large':
            self.model = PegasusForConditionalGeneration.from_pretrained('google/pegasus-large',cache_dir='./models').to(self.device)
            self.tokenizer = PegasusTokenizerFast.from_pretrained(pretrained_model_name_or_path="google/pegasus-large",cache_dir='./models')
        elif self.modelname=='pegasus-xsum' or self.modelname=='Pegasus-xsum':
            self.model = PegasusForConditionalGeneration.from_pretrained('google/pegasus-xsum',cache_dir='./models').to(self.device)
            self.tokenizer = PegasusTokenizerFast.from_pretrained(pretrained_model_name_or_path="google/pegasus-xsum",cache_dir='./models')
        elif self.modelname=='t5' or self.modelname=='T5':
            self.model = T5ForConditionalGeneration.from_pretrained("google/t5-v1_1-base",cache_dir='./models').to(self.device)
            self.tokenizer = T5TokenizerFast.from_pretrained("google/t5-v1_1-base",cache_dir='./models')
        elif 't5-large' in self.modelname or 'T5-large' in self.modelname:
            self.model = T5ForConditionalGeneration.from_pretrained("google/t5-v1_1-large",cache_dir='./models').to(self.device)
            self.tokenizer = T5TokenizerFast.from_pretrained("google/t5-v1_1-large",cache_dir='./models')
        elif self.modelname=='keybart' or  self.modelname=='KeyBART':
            self.tokenizer = AutoTokenizer.from_pretrained("bloomberg/KeyBART",cache_dir='./models')
            self.model = AutoModelForSeq2SeqLM.from_pretrained("bloomberg/KeyBART",cache_dir='./models').to(self.device)
        #self.myData = MyData
    def __token_data(self,texts,labels):
        encodings = self.tokenizer(texts, truncation=True, padding=True,model_max_length=self.len_max )
        decodings = self.tokenizer(labels, truncation=True, padding=True, model_max_length=self.len_max)
        dataset_tokenized = MyData(encodings, decodings)
        return dataset_tokenized
    def __finetune(self,freeze_encoder=None):
        prefix = 'Summary: '
        train_texts = read_text(os.path.join(self.datadir,"X.trn.txt"))
        train_index = read_index(os.path.join(self.datadir,"Y.trn.txt"))
        #train_index = list(map(lambda x: x if len(x)<10 else x[0:10],train_index))
        label_map = load_map(os.path.join(self.datadir,"output-items.txt"))
        train_labels_list = transfer_indexs_to_labels(label_map,train_index) #list,需要转化成text
        train_labels = []
        for i in train_labels_list:
            train_labels.append(", ".join(i))#是否加prefix
        train_texts = list(map(lambda x: prefix+x,train_texts))
        
        val_texts = read_text(os.path.join(self.datadir,"X.tst.txt"))
        val_index = read_index(os.path.join(self.datadir,"Y.tst.txt"))
        #val_index = list(map(lambda x: x if len(x)<10 else x[0:10],val_index))
        label_map = load_map(os.path.join(self.datadir,"output-items.txt"))
        val_labels_list = transfer_indexs_to_labels(label_map,val_index) #list,需要转化成text
        val_labels = []
        for i in val_labels_list:
            val_labels.append(",".join(i))#是否加prefix
        val_texts = list(map(lambda x: prefix+x,val_texts))
        valid_dir= self.datadir+"test_finetune.json"
        print('modelname:'+self.modelname)
        print('checkdir:'+self.checkdir)
        print('save_dir:'+self.output)
        print('batch_size:',self.batch_size)
        print('epoch:',self.epoch)
        train_dataset = []
        val_dataset = []
        for i in range(len(train_index)):
            train_dataset.append({'document':train_texts[i],'labels':train_labels[i]})
        for i in range(len(val_index)):
            val_dataset.append({'document':val_texts[i],'labels':val_labels[i]})
        from datasets import Dataset
        train_dataset = Dataset.from_list(train_dataset).shuffle(seed=44)
        val_dataset = Dataset.from_list(val_dataset).shuffle(seed=44)
        #from datasets import load_dataset
        #prefix = "summarize: "
        #dataset = load_dataset('json',data_files={'train': train_dir, 'valid': valid_dir}).shuffle(seed=42)
        #train_texts, train_labels = [prefix + each for each in train_dataset['document']], train_dataset['labels']
        #val_texts, valid_labels = [prefix + each for each in val_dataset['document']], val_dataset['labels']
        train_dataset = self.__token_data(train_texts,train_labels)
        val_dataset = self.__token_data(val_texts,val_labels)
        if freeze_encoder:
            for param in self.model.model.encoder.parameters():
                param.requires_grad = False
        train_args = Seq2SeqTrainingArguments(
            output_dir=self.checkdir,
            num_train_epochs=self.epoch,           # total number of training epochs
            per_device_train_batch_size=self.batch_size,   # batch size per device during training, can increase if memory allows
            per_device_eval_batch_size=self.batch_size,    # batch size for evaluation, can increase if memory allows
            save_steps=30000,                  # number of updates steps before checkpoint saves
            save_total_limit=5,              # limit the total amount of checkpoints and deletes the older checkpoints
            evaluation_strategy = "epoch",     # evaluation strategy to adopt during training                 # number of update steps before evaluation
            learning_rate= self.learning_rate,  # learning rate
            warmup_steps=500,                # number of warmup steps for learning rate scheduler
            weight_decay=0.01,               # strength of weight decay
            predict_with_generate=True,
        )
        self.trainer = Seq2SeqTrainer(
            model=self.model,                         # the instantiated 🤗 Transformers model to be trained
            args=train_args,                  # training arguments, defined above
            train_dataset=train_dataset,         # training dataset
            eval_dataset=val_dataset,            # evaluation dataset
            tokenizer=self.tokenizer
        )
        self.trainer.train()
        self.trainer.save_model(self.output)
        
    def train(self):
        start = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print('train start:',start)
        time_stap1 = time.process_time()
        self.__finetune()
        time_stap2 = time.process_time()
        end =  datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print('tarin end:', end)
        print('tarining cost time:'+ str((time_stap2-time_stap1)/60/60 )+"hours.")
        with open(os.path.join(self.datadir,"train_log.txt"),'a+')as w: 
            w.write("datadir:"+self.datadir+", "+"model_name: "+self.modelname+", "+"batch_size: "+str(self.batch_size)+"epoch: "+str(self.epoch)+"\n"
                    "checkdir: "+self.checkdir+", "+"output: "+self.output+"\n")
            w.write("starttime:"+start+". ")
            w.write("endtime: "+end+"\n")
    
    def __predict(self,model,tokenizer,documents):
        inputs = tokenizer(documents, return_tensors='pt', padding=True, truncation=True,model_max_length=self.len_max).to(self.device)
        #inputs = tokenizer(documents, return_tensors='pt', padding=True, truncation=True).to(device)#, padding=True
      # Generate Summary
        summary_ids = model.generate(inputs['input_ids'],max_length = 256,top_k=10,num_beams = 5).to(self.device)
        #summary_ids = model.generate(inputs['input_ids'],max_length = 256,min_length =64,num_beams = 7).to(device)  #length_penalty = 3.0  top_k = 5
        pre_result=tokenizer.batch_decode(summary_ids,skip_special_tokens=True, clean_up_tokenization_spaces=True,pad_to_multiple_of=2)
        #pred = str([tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in summary_ids])  #[2:-2]
        return pre_result   
    
    def predicting(self,modelname,src_dataname,output_dir=''):
        #modelname = os.path.join(self.datadir,modelname)
        print("modelname: ", modelname)
        #output_dir = os.path.join(self.datadir,'res',output_dir)
        print('output: '+output_dir)
        #src_dataname = os.path.join(self.datadir,src_dataname)
        print('src_data: '+src_dataname)
        #this tokenizer is not the self.tokenizer
        model = self.model
        tokenizer = self.tokenizer
        data = []
        dic = [] # dictionary for save each model generate result
        src_value = [] # using for get source document which is used to feed into model, and get predicting result
        data = read_text(src_dataname)
        res = []
        batch=[]
        # open test file 
        
        # 进度条可视化 vision process
        dataloader = DataLoader(data,batch_size= self.data_size)
        f=open(output_dir,'w+')
        f.close()
        with open(output_dir,'a+') as t:
            for i in tqdm(dataloader): #range(len(data))
                batch = i
                tmp_result = self.__predict(model,tokenizer,batch)
                for j in tmp_result:
                    l_labels = [] #l_label 是str转 label的集合
                    pre = j.strip('[]').strip().split(",")
                    for k in range(len(pre)):
                        tmpstr = pre[k].strip(" ").strip("'").strip('"')
                        if tmpstr=='':continue
                        l_labels.append(tmpstr)
                    res.append(l_labels)
                    t.write(", ".join(l_labels))
                    t.write("\n")
        return res 
        
           
    