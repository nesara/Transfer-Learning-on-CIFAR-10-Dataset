# Transfer-Learning-on-CIFAR-10-Dataset
Simple Transfer Learning on CIFAR 10 Dataset. 
It has become standard practice in computer vision tasks related to images, to use a convolutional network pre-trained as the backbone feature extraction network and train new layers on top for the target task. In this question we will implement such a model. We will use the VGG 11 bn network from the torchvision.models library as our backbone network. This model has been trained on ImageNet achieving top-5 error rate of 10.19%. It consists of 8 convolutional layers followed by adaptive average pooling and fully-connected layers to perform the classification. We will get rid of the averate pooling and fully-connected layers from the VGG 11 bn model and attach our own fully connected layers to perform the Cifar-10 classifcation.

We shall devide it into 2 parts:-

a) Instantiate a pretrained version of the VGG 11 bn model with ImageNet pre-trained weights. Add two fully
connected layers on top, with Batch Norm and ReLU layers in between them to build the CIFAR-10 10-class
classifier. Note that you will need to set the correct mean and variance in the data-loader, to match the mean
and varaince the data was normalized with when the VGG 11 bn was trained. Train only the newly added
layers while disabling gradients for rest of the network. Each parameter in PyTorch has a requires grad
flag, which can be turned off to disable gradient computation for it. Get familiar with this gradient control
mechanism in PyTorch and train the above model. As a reference point you will see validation accuracies
in the range (61-65%) if implemented correctly. 


b) We can see that while the imagNet features are useful, just learning the new layers does not yield better
performance than training our own network from scratch. This is due to the domain-shift between the
ImageNet dataset (224x224 resolution images) and the CIFAR-10 dataset (32x32 images). To improve the
performance we can fine-tune the whole network on the CIFAR-10 dataset, starting from the ImageNet
initialization. To do this, enable gradient computation to the rest of the network, and update all the model
parameters. Additionally train a baseline model where the same entire network is trained from scratch,
without loading the ImageNet weights. Compare the two models training curves, validation and testing
performance in the report.
