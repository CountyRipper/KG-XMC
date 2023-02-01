'''
This model is used to generate keyphrase with raw documents
functions: data processing, training, predicting
'''
from transformers import (BartTokenizerFast,PegasusTokenizerFast,BartForConditionalGeneration,
                          PegasusForConditionalGeneration,T5TokenizerFast,T5ForConditionalGeneration,
                          AutoTokenizer,AutoModel)
import torch
from torch.utils.data import DataLoader
import wandb
import pytorch_lightning as pl
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

class kg_model(pl.LightningModule):
    def __init__(self,args) -> None:
        self.model_name = args.kg_modelname 
        self.batch_size = args.kg_batch_size
        self.lr = args.kg_lr
        self.epoch = args.kg_epoch
        self.type= args.kg_type
        self.datadir = args.datadir
        self.save_dir = args.kg_savedir
        self.curr_avg_loss = 0.0
        if self.type in 'bart':
            self.model = BartForConditionalGeneration.from_pretrained(self.model_name).to(device)
            self.tokenizer = BartTokenizerFast.from_pretrained(self.model_name)
        if self.type in 'pega':
            self.model =  PegasusForConditionalGeneration.from_pretrained(self.model_name).to(device)
            self.tokenizer = PegasusTokenizerFast.from_pretrained(self.model_name)
        if self.type in 't5':
            self.model =  T5ForConditionalGeneration.from_pretrained(self.model_name).to(device)
            self.tokenizer = T5TokenizerFast.from_pretrained(self.model_name)
        else:
            self.model  = AutoModel.from_pretrained(self.model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        # forward part
        def forward(self, encoder_input_ids, decoder_input_ids):
            return self.model(input_ids=encoder_input_ids, labels=decoder_input_ids)
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
            res = self(encoder_input_ids, labels)
            #loss, logits = self(encoder_input_ids,labels)

            self.log('val_loss', res.loss, prog_bar=True,batch_size=self.hparameters['batch_size'])
            return res.loss

        def configure_optimizers(self):
            optimizer = torch.optim.Adam(self.parameters(), lr=self.hparameters['learning_rate'])
            
            return{
                "optimizer": optimizer,
                #"lr_scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=100),
                "lr_scheduler":torch.optim.lr_scheduler.StepLR(optimizer=optimizer,step_size=1,gamma=0.999),
                "interval": "step",
                "frequency": 1,
            }
        def generate(self, input_ids, attention_mask, max_length, top_k, num_beams):
            return self.model.generate(input_ids=input_ids, attention_mask=attention_mask,
                                       max_length=max_length, top_k=top_k, num_beams=num_beams)
