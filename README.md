# AnimeFace

Sample code of [Chainer](http://chainer.org/).

## Description
We can training and recognize anime character face.
We use [animeface-character-dataset](http://www.nurs.or.jp/~nagadomi/animeface-character-dataset/) for training.
I refered [深層学習でアニメ顔を分類する](http://qiita.com/hogefugabar/items/312707a09d29632e7288) when I make these program.

## Demo

## Usage

### Training
Build Dockerfile.

```
docker build -t <image_name> .
```

Run training.

```
docker run <image_name>
```

Then `/animeface/animeface.model` will be created in container.

### Recognition
Run command in container.

```
python recognize.py <path_to_image_file>
```

`animeface.model` is necessary to recognize.

## Author

[shibataka000](https://github.com/shibataka000)
