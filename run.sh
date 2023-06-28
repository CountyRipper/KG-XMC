#! /bin/zsh
#conda activate ddd
#nohup python -u ./src/main_run.py --datadir='./dataset/Wiki10-31K/' --istrain=1 --is_pred_trn=1 --is_pred_tst=1 --iscombine=1 --is_rank_train=1 --is_ranking=1 --combine_model='cross-encoder/stsb-roberta-base' --modelname='t5' --outputmodel='t5_save' --batch_size=8 --epoch=10 --checkdir='t5_check' --data_size=4  --rank_model='all-MiniLM-L6-v2' --rank_batch=128 --rankmodel_save='bi_en_t5'>> ./log/output.log 2>&1 &
python -u ./main.py --datadir='./dataset/wiki-500k' \
--is_kg_train=1 \
--is_kg_pred_trn=1 \
--is_kg_pred_tst=0 \
--is_combine=1 \
--is_rank_train=1  \
--is_rank=1 \
--kg_type='bart' \
--combine_type='bi' \
--rank_type='bi' \
--kg_epoch=4 \
--kg_batch_size=4 \
--kg_checkdir='bart_check' \
--kg_savedir='bart_save' \
--kg_lr=2e-5 \
--data_size=16 \
--combine_model_name='sentence-transformers/all-MiniLM-L12-v2' \
--rank_model='sentence-transformers/all-MiniLM-L12-v2' \
--rank_batch=64 \
--rank_epoch=3 \
--rankmodel_save='bi_rank'

