# Hashtag Prediction

Hashtag Prediction is a novel predictor that predict hastag from instagram image dataset. It uses [HARRISON Benchmark Dataset] from https://github.com/minstone/HARRISON-Dataset as the dataset. Different from the way they train, our model will uses inception model that have lower number of parameters and the label will be treated as having relation, similar to our implementation of [RecSys=DAE]. Depending on the result of the model, an android application will also be made if the execution time is on the tolerable limit.

### Tech

Hashtag Prediction uses a number of open source projects to work properly:

* [Tensorflow] - The best, hassle free deep learning framework
* [Slim] - The higher level representation of Tensorflow
* [Pandas] - Library to read data

### Installation

Install the dependencies then execute these commands.

```sh
$ git clone https://github.com/ardiya/HashtagPrediction.git
$ cd HashtagPrediction
$ python preprocess.py
$ python main.py
```

Readmes, how to use them can be found here:

* [README.md] [PlDb]

### Done

 - Convert images into TFRecords and read the records

### Todos

 - Improve CNN model using inception
 - Load the weight from trained inception model
 - Provide MAP measurement
 - Compare with 

License
----

MIT


**Free Software, Hell Yeah!**

[//]: # (These are reference links used in the body of this note and get stripped out when the markdown processor does its job. There is no need to format nicely because it shouldn't be seen. Thanks SO - http://stackoverflow.com/questions/4823468/store-comments-in-markdown-syntax)
[HARRISON Benchmark Dataset]: <https://github.com/minstone/HARRISON-Dataset>
[RecSys=DAE]:<https://github.com/ardiya/RecSys-DAE-tensorflow>
[Tensorflow]:<https://tensorflow.org>
[Slim]:<https://github.com/tensorflow/models/blob/master/inception/inception/slim/README.md>
[Pandas]:<pandas.pydata.org/>
[PlDb]: <https://github.com/ardiya/HashtagPrediction/README.md>
   