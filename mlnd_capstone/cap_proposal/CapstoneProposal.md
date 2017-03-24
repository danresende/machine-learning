# Machine Learning Engineer Nanodegree
## Capstone Proposal
Daniel Resende  
March 24th, 2017

## Proposal

### Domain Background

Image recognition is a deeply studied field due to its vast applicability, e.g. accessibility with image caption, readding document, etc. With that in mind, recognizing characters became a well attacked challenge, thus datasets like the MNIST([1](http://yann.lecun.com/exdb/mnist/)) are thoroughly addressed with algorithms approching perfect performance by now.

But recognizing characters in real-world image imposes a lot more difficult due to the variation and distractions as it can be seen in papers addressing this kinds of datasets like Multi-digit Number Recognittion from Street View Imagery using Deep Convolutional Neural Networks([2](https://arxiv.org/abs/1312.6082)) and Very Deep Convolutional Networks for Large-Scale Image Recognition([3](https://arxiv.org/abs/1409.1556)) and others. 

### Problem Statement

With the MNIST([1](http://yann.lecun.com/exdb/mnist/)) dataset we can see how a character recognition problem can be addressed in a document clean like dataset. On the other hand in Multi-digit Number Recognittion from Street View Imagery using Deep Convolutional Neural Networks([2](https://arxiv.org/abs/1312.6082)), the authors used the Street View House Number dataset as a representation of real-world image to understand the difficults imposed by this kind of dataset. Thus the problem I'll try to addressin with this project is going to be a mixture of both.

I'll use the MNIST-like format of the Street View House Number([4](http://ufldl.stanford.edu/housenumbers/)) dataset to give a real-world challenge to a now well know problem, character recognition, and see if it can achive at least human-level accuracy with this dataset.

### Datasets and Inputs

So in order to attack and understand this real-world problem I'll use the Street View House Number([4](http://ufldl.stanford.edu/housenumbers/)) dataset, a benchmark dataset introduced by Yuval Netzer et al.([5](http://ufldl.stanford.edu/housenumbers/nips2011_housenumbers.pdf)) that comes in two flavors: (1) with original full numbers, variable resolution color house-number; and (2) in a character level MNIST-like format resized to a fixed 32-by-32 pixels resolution.

I will use the MNIST-like of the Street View House Number([2](http://ufldl.stanford.edu/housenumbers/)) dataset, because it can be consider is the next step from the MNIST([1](http://yann.lecun.com/exdb/mnist/)) dataset. This is an open dataset that have been taken from the original house numbers character bounding boxes and extended in the appropriate dimension to become square windows, so that resizing them to 32-by-32 pixels does not introduce aspect ratio distortions. But still this preprocessing introduces some distracting digits to the sides of the digit of interest.

### Solution Statement

Since Yann LeCunn's Le Net, deep learning has been established as a go to way of handling image recognition problems. Thus we are going to use a deep convolutional neural network architecture known as VGG16 to see if it can surpass the distractions and difficults of the real-world characters presented in the dataset.

Simonyan and Zisserman state that incresing depth using an architecure with very small convolution filters can significantly improve the results comparing to smaller versions, resulting in the winning of the ImageNet Challenge 2014. Thus it is expectable that this model can be effective.

### Benchmark Model

Multi-digit Number Recognittion from Street View Imagery using Deep Convolutional Neural Networks([2](https://arxiv.org/abs/1312.6082)), Goodfellow et al. achieve over 96% accuracy in recognizing complete street number digit in the same Street View dataset. They also realised that performance increase with deeper networks and to achieve the mentioned accuracy they used a eleven layer network.

Thus I expect that the VGG16 will performe as good as the one used in the Multi-digit paper being able to come around the noise created by the resizing of the images as mentioned before.

### Project Design

To put in a better perspective this project will require downloading the data from the dataset website first. I'll use only the `train_32x32.mat`, because this file has already 73,257 images what I belive is more than enough to do this project. This file is a *.mat* file, to use it with Python will read it using the Scipy module.

![Street View House Number Dataset](http://ufldl.stanford.edu/housenumbers/32x32eg.png)

Those images are, as said before, a 32-by-32 pixel color images with pixels ranging from 0 to 255 for each color, thus will need to normalize the data before using our neural network putting the pixel values between 0 to 1. Also the labels come in a numerical form ranging from 1 to 10, with 10 being 0 in the picture. So I'll one-hot enconde the labels so our model can predict the probability of an given image be of a number.

For better generalization, I'll split the data in three parts: train, validation and test. The first I'll user to train the data and find the optimal weights validating along the way with the validation set. Finnaly after training I'll use the test data to see if the model was able to learn and generalize to images that it never have seen before. This step is very important because the model can easyly overfit and learn only how to recognize the train data. Keeping in mind that with 10% accuracy the model is just randomly guessing and not learning at all. 

To recognize the image's number I'll use the VGG16 architecture, that consist in a 13 layer convolutional neural network with 3 fully connected neural network. For the desing of this architecture I will use Tensorflow[6](https://www.tensorflow.org), building first the nodes that I'll be using to assemble the graph and, then, build the architecture.

![VGG16 architecure](https://heuritech.files.wordpress.com/2016/02/vgg16.png?w=470)

To compute the loss and give the model tha ability to learn, I will use softmax with an Adam optimizer, an algorithm for first-order gradient-based optimization of stochastic objective functions, based on adaptive estimates of lower-order moments([9](https://arxiv.org/abs/1412.6980)).

Finally, will test it to see how well it does with new unseen images, hopping to get an high accuracy with this technic.

### References

1. The MNIST Database of handwritten digits - [http://yann.lecun.com/exdb/mnist/](http://yann.lecun.com/exdb/mnist/)
2. GOODFELLOW, I. J. et al. - Multi-digit Number Recognittion from Street View Imagery using Deep Convolutional Neural Networks - [https://arxiv.org/abs/1312.6082](https://arxiv.org/abs/1312.6082)
3. SIMONYAN, K. and ZISSERMAN, A. - Very Deep Convolutional Networks for Large-Scale Image Recognition - [https://arxiv.org/abs/1409.1556](https://arxiv.org/abs/1409.1556)
4. The Street View House Numbers (SVHN) Dataset - [http://ufldl.stanford.edu/housenumbers/](http://ufldl.stanford.edu/housenumbers/)
5. NETZER, Y. et al. - Reading Digits in Natural Images with Unsupervised Feature Learning - [http://ufldl.stanford.edu/housenumbers/nips2011\_housenumbers.pdf](http://ufldl.stanford.edu/housenumbers/nips2011_housenumbers.pdf)
6. Tensorflow.org - [https://www.tensorflow.org](https://www.tensorflow.org)
7. VGG in TensorFlow - [https://www.cs.toronto.edu/~frossard/post/vgg16/](https://www.cs.toronto.edu/~frossard/post/vgg16/)
8. A Brief Report Of The Heuritech Deep Learning Meetup #5 - [https://blog.heuritech.com/2016/02/29/a-brief-report-of-the-heuritech-deep-learning-meetup-5/](https://blog.heuritech.com/2016/02/29/a-brief-report-of-the-heuritech-deep-learning-meetup-5/)
9. Adam: A Method for Stochastic Optimization - [https://arxiv.org/abs/1412.6980](https://arxiv.org/abs/1412.6980)