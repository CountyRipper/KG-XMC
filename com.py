from src.utils.premethod import load_map,p_at_k
import os
from src.model.rank_model import label_rank

label_rank(src_labels='./dataset/eurlex-4k/res/tst_xr.txt',
           src_text='./dataset/eurlex-4k/X.tst.txt',model_type='bi',
           model_name='./dataset/eurlex-4k/bi_rank',
           label_map='./dataset/eurlex-4k/output-items.txt',
           output_text='./dataset/eurlex-4k/res/tst_concat_rank.txt',
           output_index='./dataset/eurlex-4k/res/tst_concat_rank_index.txt',
           )
p_at_k(dir='./dataset/eurlex-4k/',src_label_dir=os.path.join('./dataset/eurlex-4k/','Y.tst.txt'),
       pred_label_dir='./dataset/eurlex-4k/res/tst_concat_rank_index.txt',outputdir='./dataset/eurlex-4k/res_concate.txt')