'''
This model is used to generate keyphrase with raw documents
functions: data processing, training, predicting
'''
import os
from transformers import (BartTokenizerFast,PegasusTokenizerFast,BartForConditionalGeneration,
                          PegasusForConditionalGeneration,T5Tokenizer,T5TokenizerFast,T5ForConditionalGeneration,
                          AutoTokenizer,AutoModel,BartTokenizer,MBartForConditionalGeneration, MBart50TokenizerFast,
                          GPT2Tokenizer, GPT2Model
)
import torch
from torch.utils.data import DataLoader
import wandb
import pytorch_lightning as pl
from tqdm import tqdm
from pytorch_lightning.callbacks import EarlyStopping,ModelCheckpoint,LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from src.utils.premethod import load_map, read_index, read_text, transfer_indexs_to_labels
wandb.init(project="kg_model")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
class MyData(torch.utils.data.Dataset):
    def __init__(self, encoding, labels):
        self.ids = encoding['input_ids']
        self.mask = encoding['attention_mask']
        self.labels = labels['input_ids']
    def __getitem__(self, idx):
        item = {}
        item['input_ids'] = torch.tensor(self.ids[idx]).to(device)
        item['attention_mask'] = torch.tensor(self.mask[idx]).to(device)
        item['labels'] = torch.tensor(self.labels[idx]).to(device)
        #item={'input_ids': torch.tensor(val[idx]).to(device) for key, val in self.encoding.items()}
        #item['labels'] = torch.tensor(self.labels['input_ids'][idx]).to(device)
        return item

    def __len__(self):
        return len(self.labels)  # len(self.labels)

class KG_Model(pl.LightningModule):
    def __init__(self,args) -> None:
        super().__init__()
        self.model_name = args.kg_modelname 
        self.batch_size = args.kg_batch_size
        self.lr = args.kg_lr
        #self.epoch = args.kg_epoch
        self.type= args.kg_type
        self.datadir = args.datadir
        self.trn = args.kg_trn_data
        self.tst = args.kg_tst_data
        self.top_k = args.top_k
        self.top_p = None
        if args.top_p:
            self.top_p= args.top_p
        if args.max_len:
            self.max_len = args.max_len
        #self.save_dir = args.kg_savedir
        self.curr_avg_loss = 0.0
        if 'pega' in self.type :
            self.model =  PegasusForConditionalGeneration.from_pretrained(self.model_name).to(device)
            self.tokenizer = PegasusTokenizerFast.from_pretrained(self.model_name,model_max_length = self.max_len)
        elif 'mbart' in self.model_name:
            self.model = MBartForConditionalGeneration.from_pretrained(self.model_name).to(device)
            self.tokenizer = MBart50TokenizerFast.from_pretrained(self.model_name,model_max_length = self.max_len)
        elif 'bart'in self.type :
            self.model = BartForConditionalGeneration.from_pretrained(self.model_name).to(device)
            self.tokenizer = BartTokenizerFast.from_pretrained(self.model_name,model_max_length = self.max_len)
        elif 't5' in self.type :
            self.model =  T5ForConditionalGeneration.from_pretrained(self.model_name).to(device)
            self.tokenizer = T5TokenizerFast.from_pretrained(self.model_name,model_max_length = self.max_len)
        elif 'gpt2' in self.model_name:
            self.model = GPT2Model.from_pretrained(self.model_name)
            self.tokenizer = GPT2Tokenizer.from_pretrained(self.model_name,model_max_length = self.max_len).to(device)
        else:
            print('未识别model')
            self.model  = AutoModel.from_pretrained(self.model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        self.save_hyperparameters()
        # forward part
    def forward(self, encoder_input_ids, labels):
        return self.model(input_ids=encoder_input_ids,labels=labels)
    def training_step(self, batch, batch_idx):
        encoder_input_ids, encoder_attention_mask, labels = torch.stack([i['input_ids'] for i in batch]), torch.stack(
            [i['attention_mask'] for i in batch]), torch.stack([i['labels'] for i in batch])
        res = self(encoder_input_ids, labels)
        # loss = self.custom_loss(logits, labels) custom loss
        cur_lr = self.trainer.optimizers[0].param_groups[0]['lr']
        global_step = self.trainer.global_step
        # #手动优化scheduler
        sch = self.lr_schedulers()
        loss = res.loss
        self.curr_avg_loss+=loss
        if (global_step+1)%50 == 0:
            wandb.log({"loss": self.curr_avg_loss/50,"global_step":global_step})
            wandb.log({"learning_rate":cur_lr,"global_step":global_step})
            wandb.log({"train_epoch":self.trainer.current_epoch,"global_step":global_step})
            self.curr_avg_loss= 0.0
        if (batch_idx + 1) % 5 == 0:
            sch.step()
        self.log('lr',cur_lr, prog_bar=True, on_step=True)
        #self.log('train_loss', loss, prog_bar=True,batch_size=self.hparameters['batch_size'])
        return loss
    def validation_step(self, batch, batch_idx):
        encoder_input_ids, encoder_attention_mask, labels = torch.stack([i['input_ids'] for i in batch]), torch.stack(
            [i['attention_mask'] for i in batch]), torch.stack([i['labels'] for i in batch])
        #encoder_input_ids, encoder_attention_mask,labels = batch['input_ids'],batch['attention_mask'],batch['labels']
        res = self(encoder_input_ids=encoder_input_ids, labels=labels)
        #loss, logits = self(encoder_input_ids,labels)
        self.log('val_loss', res.loss, prog_bar=True,batch_size=self.batch_size)
        return res.loss
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        
        return{
            "optimizer": optimizer,
            #"lr_scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=100),
            "lr_scheduler":torch.optim.lr_scheduler.StepLR(optimizer=optimizer,step_size=1,gamma=0.9995),
            "interval": "step",
            "frequency": 1,
        }
    def generate(self, input_ids, attention_mask, max_length,num_beams): 
        if self.top_p and self.top_k:
               return self.model.generate(input_ids=input_ids, attention_mask=attention_mask,
                                   max_length=max_length, num_beams=num_beams,top_p=self.top_p)
        else:
            return self.model.generate(input_ids=input_ids, attention_mask=attention_mask,
                                   max_length=max_length, num_beams=num_beams)
    def train_dataloader(self):
        datadir = self.datadir
        prefix = "Summary: "
        #获取text, 获取label index，映射出label text
        train_texts = read_text(os.path.join(datadir,"X.trn.txt"))
        train_index = read_index(os.path.join(datadir,self.trn))
        #train_index = list(map(lambda x: x if len(x)<10 else x[0:10],train_index))
        label_map = load_map(os.path.join(datadir,"output-items.txt"))
        train_labels_list = transfer_indexs_to_labels(label_map,train_index) #list,需要转化成text
        train_labels = []
        for i in train_labels_list:
            train_labels.append(", ".join(i))#是否加prefix
        train_texts = list(map(lambda x: prefix+x,train_texts))
        # for i in range(len(train_texts)):
        #     train_texts[i] = prefix+train_texts[i]
        '''
        在这里加入shuffle或者sort,改变train_labels
        '''
        
        encodings = self.tokenizer(train_texts, truncation=True, padding=True)
        encodings = self.tokenizer(train_texts, truncation=True, padding=True)
        decodings = self.tokenizer(train_labels, truncation=True, padding=True)
        dataset_tokenized = MyData(encodings, decodings)
        train_data = DataLoader(
            dataset_tokenized, batch_size=self.batch_size, collate_fn=lambda x: x, shuffle=True)
        # create a dataloader for your training data here
        return train_data
    def val_dataloader(self):
        datadir = self.datadir
        prefix = "Summary: "
        #获取text, 获取label index，映射出label text
        val_texts = read_text(os.path.join(datadir,"X.tst.txt"))
        val_index = read_index(os.path.join(datadir,self.tst))
        #val_index = list(map(lambda x: x if len(x)<10 else x[0:10],val_index))
        label_map = load_map(os.path.join(datadir,"output-items.txt"))
        val_labels_list = transfer_indexs_to_labels(label_map,val_index) #list,需要转化成text
        val_labels = []
        for i in val_labels_list:
            val_labels.append(",".join(i))#是否加prefix
        val_texts = list(map(lambda x: prefix+x,val_texts))
        '''
        在这里加入shuffle或者sort,改变train_labels
        '''
        encodings = self.tokenizer(val_texts, truncation=True, padding=True)

        encodings = self.tokenizer(val_texts, truncation=True, padding=True)
        decodings = self.tokenizer(val_labels, truncation=True, padding=True)
        dataset_tokenized = MyData(encodings, decodings)
        val_data = DataLoader(
            dataset_tokenized, batch_size=self.batch_size, collate_fn=lambda x: x, shuffle=True)
        # create a dataloader for your training data here
        return val_data

def get_predict(datadir,documents,tokenizer,model):
    # constraint_list = load_map(os.path.join(datadir,'output-items.txt'))
    # for i in constraint_list:
    #     tokenizer.unique_no_split_tokens.append(i)
    # Constraint = tokenizer(constraint_list,add_special_tokens=False).input_ids
    inputs = tokenizer(documents, return_tensors='pt', padding=True, truncation=True).to(device)
    #print('constraint\n')
    summary_ids = model.generate(inputs['input_ids'],inputs['attention_mask'], max_length = 128,num_beams = 5).to(device)
    pre_result=tokenizer.batch_decode(summary_ids,skip_special_tokens=True, clean_up_tokenization_spaces=True,pad_to_multiple_of=2)
    return pre_result

def kg_predict(model,datadir,src_dir,output_dir,data_size,model_type='bart'):
    '''
    model is the trained model of pretrained text2text model like BART.
    tokenizer is the tokenizer which is compaitible for model.
    src is the complete relateive dir of documents Y.tst.txt.
    outputdir is the output dir which is in the datadir.
    datasize is the batch_decode size.
    '''
    print(f'src_dir: {src_dir}')
    print(f'output_dir: {output_dir}')
    res=[]
    doc_list=[]
    with open(src_dir,'r+') as f:
        for i in f:
            doc_list.append(i)
    dataloader = DataLoader(doc_list,batch_size=data_size)
    tokenizer = model.tokenizer
    
    with open(output_dir,'w+') as t:
        for i in tqdm(dataloader): #range(len(data))
            tmp_result = get_predict(datadir=datadir,documents=i,tokenizer=tokenizer,model=model)
            for j in tmp_result:
                l_labels = [] #l_label 是str转 label的集合
                pre = j.replace("Summary: ","").strip().split(", ")
                for k in range(len(pre)):
                    tmpstr = pre[k].strip(" ").strip("'").strip('"')
                    if tmpstr=='':continue
                    l_labels.append(tmpstr)
                res.append(l_labels)
                t.write(",".join(l_labels))
                t.write("\n")



def kg_train(args):
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, mode='min')
    
    checkpoint_callback = ModelCheckpoint(
            dirpath='./log/kg_check',
            filename='{epoch}-{val_loss:.2f}-{other_metric:.2f}'
        )
    lr_callback = LearningRateMonitor(logging_interval="step")
    model = KG_Model(args)
    #tmp = model.train_dataloader()
    #logger = TensorBoardLogger(save_dir=os.path.join(hparams['data_dir'],'t2t'),name=hparams['model']+'_log')
    
    # if not os.path.exists(args.kg_savedir):
    #     os.mkdir(args.kg_savedir)
    trainer = pl.Trainer(max_epochs=args.kg_epoch, callbacks=[early_stopping,checkpoint_callback ,lr_callback],
                         #auto_lr_find=True,
                         #default_root_dir=os.path.join(args.kg_savedir),
                         accelerator="gpu", devices=1)
    
    trainer.fit(model,train_dataloaders=model.train_dataloader(),
                    val_dataloaders=model.val_dataloader())
    print(f'save model in {os.path.join(args.datadir,args.kg_savedir)}')
    trainer.save_checkpoint(os.path.join(args.datadir,args.kg_savedir))
    return os.path.join(args.datadir,args.kg_savedir)