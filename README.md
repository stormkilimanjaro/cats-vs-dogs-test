# cats-vs-dogs-test
Attempting to build a convolutional neural network to test [Kaggle's Cats vs Dogs](https://www.kaggle.com/c/dogs-vs-cats) dataset 

**References**:
* [Building a Cat Detector using Convolutional Neural Networks — TensorFlow for Hackers (Part III)](https://medium.com/@curiousily/tensorflow-for-hackers-part-iii-convolutional-neural-networks-c077618e590b)
* [Classifying Cats vs Dogs with a Convolutional Neural Network on Kaggle](https://pythonprogramming.net/convolutional-neural-network-kats-vs-dogs-machine-learning-tutorial/)

## Before running the script

### Renaming the files
* Download the Kaggle dataset and rename the images:
  
  All the pictures in the `Cat` folder should have the `Cat` prefix with a number, eg, `Cat0.jpg`
  
  All pictures in the `Dog` folder shoud have the `Dog` prefix, eg, `Dog0.jpg`
* Copy all the renamed files into the folder `train`
* Cut 250 cat images and 250 dog images into the folder `test`

### Ensure the following packages are installed:
* [Tensorflow](https://www.tensorflow.org)
* [NumPy](http://www.numpy.org)
* [Tflearn](http://tflearn.org)
* [OpenCV](https://opencv.org/)
* [Tqdm](https://github.com/tqdm/tqdm)

## How to run:

In the directory with the `train` and `test` sub-directories, run:

```
python process.py
```

