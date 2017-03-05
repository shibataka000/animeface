# AnimeFace

Sample code of [Chainer](http://chainer.org/).

## Description
We can training and recognize anime character face.
We use [animeface-character-dataset](http://www.nurs.or.jp/~nagadomi/animeface-character-dataset/) for training.
I refered [深層学習でアニメ顔を分類する](http://qiita.com/hogefugabar/items/312707a09d29632e7288) when I make these program.

## Usage

### Training
Run command in container.

```
./train.sh
```

Then `animeface.model` will be created in container.

### Recognition
Run command in container.

```
./recognize.sh <path_to_image_file>
```

`animeface.model` is necessary to recognize.
Recognized image must be in container.

## Install
Build Dockerfile.

```
docker build -t <image_name> .
```

## Author

[shibataka000](https://github.com/shibataka000)
