from argparse import ArgumentParser
from run import run
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--datadir', type=str, default='./dataset/eurlex-4k/',
                        help='dataset_dir')#eurlex-4k, amazoncat-13k,wiki10-31k,wiki500k,amazon-3m,amazon-670k
    parser.add_argument('--kg_modelname', type=str,default='facebook/bart-base',
                        help='kg_modelname')
    parser.add_argument('--kg_type',type=str,default='bart')#bart,pega,t5
    parser.add_argument('--combine_type',type=str,default='bi')# bi, cr, del, sim
    parser.add_argument('--rank_type',type=str,default='bi')# bi,cr,sim
    parser.add_argument('--kg_sw',type=str,default='pl')# pl, hg
    parser.add_argument('--max_len',type=int,default=1024) #''tokenizer max length of document.
    # finetune args
    parser.add_argument('--is_kg_train',type=int,default=1,
                        help="whether run finteune processing")
    
    parser.add_argument('-b', '--kg_batch_size', type=int, default=4,
                        help='number of batch size for training')
    parser.add_argument('-e', '--kg_epoch', type=int, default=3,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--kg_checkdir', type=str, default='bart_check',
                        help='path to trained model to save')
    parser.add_argument('--kg_savedir',type=str,default='bart_save',
                        help="fine-tune model save dir")
    parser.add_argument('--kg_lr', type=float, default=2e-5,
                        help='learning rate')
    parser.add_argument('--kg_seed', type=int, default=44,
                        help='random seed (default: 1)')
    parser.add_argument('--kg_trn_data', type=str, default='Y.trn.txt')
    parser.add_argument('--kg_tst_data', type=str, default='Y.tst.txt')
    #perdicting args
    parser.add_argument('--is_kg_pred',type=int,default=1, help="whether predict")
    parser.add_argument('--is_kg_pred_trn',type=int,default=1,
                        help="Whether run predicting training dataset")
    parser.add_argument('--is_kg_pred_tst',type=int,default=1,
                        help="Whether run predicting testing dataset")
    parser.add_argument('--top_k',type=int,default=10)
    parser.add_argument('--data_size',type=int,default=12)
    parser.add_argument('--top_p',type=float,default=0.75)
    #combine part
    parser.add_argument('--is_combine',type=int,default=1,
                        help="Whether run combine")
    parser.add_argument('--combine_model_name',type=str,default='sentence-transformers/all-MiniLM-L12-v2')

    #rank part
    parser.add_argument('--is_rank_train',type=int,default=1)
    parser.add_argument('--rank_model',type=str,default='sentence-transformers/all-MiniLM-L12-v2')
    parser.add_argument('--rank_batch',type=int,default=64)
    parser.add_argument('--rank_epoch',type=int,default=3)
    parser.add_argument('--rankmodel_save',type=str,default='bi_rank')
    parser.add_argument('--is_rank',type=int,default=1)
    args = parser.parse_args()
    #args.datadir = './dataset/wiki10-31k/'
    #args.kg_sw = 'hg'
    #args.is_kg_train=0
    #args.is_kg_pred = []
    #args.is_kg_pred_trn=0
    #args.is_kg_pred_tst=0
    #args.is_combine=0
    #args.is_rank_train=0
    #args.is_rank=0
    # args.combine_model='bi-encoder'
    #args.combine_model_name='all-MiniLM-L6-v2'
    # args.modelname='bart'
    # args.outputmodel='bart_save'
    # args.batch_size=4
    # args.t2t_epoch=3
    # args.t2t_lr=5e-5
    # args.checkdir='bart_check'
    # args.data_size=4
    # args.rank_model='all-MiniLM-L12-v2'
    # args.rank_batch=64
    # args.rank_epoch= 3
    # args.rankmodel_save='t2t_bi_bi64'
    # args.rank_is_trained = 0
    run(args)