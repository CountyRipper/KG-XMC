#!/bin/sh 
DATASET= ""
if [ $# == 0 ];then
    DATASET="eurlex-4k"
else
    DATASET=$1
fi
echo ${DATASET}

# 下载数据集downloading dataset
wget -nv -nc https://archive.org/download/pecos-dataset/xmc-base/${DATASET}.tar.gz -P ./dataset/
#解压 uncompress tar file
tar -zvxf ./dataset/${DATASET}.tar.gz -C ./dataset/
# file归档
mv  ./dataset/xmc-base/${DATASET} ./dataset
rm -r ./dataset/xmc-base
rm ./dataset/${DATASET}.tar.gz
