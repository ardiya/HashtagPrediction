# Hashtag Prediction

Hashtag Prediction is a novel predictor that predict hastag from instagram image dataset. It uses [HARRISON Benchmark Dataset] from https://github.com/minstone/HARRISON-Dataset as the dataset. Different from the way they train, our model will uses inception model that have lower number of parameters and the label will be treated as having relation, similar to our implementation of [RecSys-DAE]. Depending on the result of the model, an android application will also be made if the execution time is on the tolerable limit.

### Tech

Hashtag Prediction uses a number of open source projects to work properly:

* [Tensorflow] - The best, hassle free deep learning framework
* [Slim] - The higher level representation of Tensorflow
* [Pandas] - Library to read data

### Installation

Installing latest version of TF-slim

As of 8/28/16, the latest stable release of TF is r0.10, which contains most of TF-Slim but not some later additions. To obtain the latest version, you must install the most recent nightly build of TensorFlow. You can find the latest nightly binaries at [TensorFlow Installation] in the section that reads "People who are a little more adventurous can also try our nightly binaries". Copy the link address that corresponds to the appropriate machine architecture and python version, and pip install it. For example:

```sh
export TF_BINARY_URL=https://ci.tensorflow.org/view/Nightly/job/nightly-matrix-cpu/TF_BUILD_CONTAINER_TYPE=CPU,TF_BUILD_IS_OPT=OPT,TF_BUILD_IS_PIP=PIP,TF_BUILD_PYTHON_VERSION=PYTHON2,label=cpu-slave/lastSuccessfulBuild/artifact/pip_test/whl/tensorflow-0.10.0rc0-cp27-none-linux_x86_64.whl
sudo pip install --upgrade $TF_BINARY_URL
```

Install those dependencies then execute these commands.

```sh
$ git clone https://github.com/ardiya/HashtagPrediction.git
$ cd HashtagPrediction
$ python preprocess.py
$ python train.py
$ python evaluate.py
```

Readmes, how to use them can be found here:

* [README.md] [PlDb]

### Done

 - Split dataset into 52373 train data and 5000 test data
 - Convert images into TFRecords and read the records
 - Use Inception V3
 - Load the trained weight
 - Provide MAP measurement

### Todos

 - Compare with benchmark
 - Implement Android Application

License
----

MIT


[//]: # (These are reference links used in the body of this note and get stripped out when the markdown processor does its job. There is no need to format nicely because it shouldn't be seen. Thanks SO - http://stackoverflow.com/questions/4823468/store-comments-in-markdown-syntax)
[HARRISON Benchmark Dataset]: <https://github.com/minstone/HARRISON-Dataset>
[RecSys-DAE]:<https://github.com/ardiya/RecSys-DAE-tensorflow>
[Tensorflow]:<https://tensorflow.org>
[TensorFlow Installation]:<https://github.com/tensorflow/tensorflow#installation>
[Slim]:<https://github.com/tensorflow/models/blob/master/inception/inception/slim/README.md>
[Pandas]:<pandas.pydata.org/>
[PlDb]: <https://github.com/ardiya/HashtagPrediction/README.md>
   