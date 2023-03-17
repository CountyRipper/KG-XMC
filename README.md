# KG-XMC
a standard XMC solution repository by KG method

## Data prosessing

## dataset

X.txt:文章raw text \
Y.npz:文章的label index --> 稀疏矩阵spacy\
output-items.txt：index和实体标签对应文件\
tfidf-attnxml/X.npz:  文章的tfidf向量

how to access dataset:
In the top dir(terminal) \
runing 
`data.sh wiki10-31k`
shellscript can download dataset files and unzip them automatically.
## Model overivew

## pipeline
after runing `data.sh` \
runing  `run.sh` \
run.sh script can control pipeline and hyperparameters.

## Git
Github link is `https://github.com/CountyRipper/KG-XMC.git`


