#!/bin/sh
#curl http://www.nurs.or.jp/~nagadomi/animeface-character-dataset/data/animeface-character-dataset.zip -o animeface-character-dataset.zip
curl https://s3-ap-northeast-1.amazonaws.com/shibataka000-myawsbucket/animeface-character-dataset.zip -o animeface-character-dataset.zip
unzip animeface-character-dataset.zip
python train.py
