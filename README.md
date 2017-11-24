# FacialRecognition
Facial Expression Recognition with TensorFlow


## Introduction
* Facial Expression Recognition with Deep-Learning.
* Implementation CNN(Convolutional Neural Network) with _TensorFlow 1.4_.


## Codes
* Test_Images : Directory of images for testing model.
* Train_Images : Directory of images for traning neural-network.
* collect_images.py : Collect face images from _Bing_ and _Google_.
* convert_images.py : Convert images files(*.jpg, *.jpeg, *.png) to dataset file(*.bin).
* dataset.py : Dataset class for training or testing neural-network.
* cnn.py : Create CNN and train them or classify images.


## Example of Run Codes
* Convert images to dataset
```
>>> import convert_images as ci

>>> ci.IMAGES_DIR = './Train_Images'
>>> ci.main('./train.bin', shuffle=True)
```

* Train CNN and save model
```
>>> from cnn import Cnn

>>> my_cnn = Cnn()

>>> my_cnn.set_device('gpu')
>>> my_cnn.set_epoch(1000)
>>> my_cnn.set_batch_size(100)

>>> my_cnn.train('./train.bin', './CNN_Models/model')

>>> del my_cnn
```

* Evaluate saved model
```
>>> from cnn import Cnn

>>> my_cnn = Cnn()

>>> my_cnn.set_device('gpu')
>>> my_cnn.set_batch_size(100)

>>> result = my_cnn.eval('./test.bin', './CNN_Models/model')
>>> print(result)

>>> del my_cnn
```

* Classify label of new images
```
>>> from cnn import Cnn

>>> my_cnn = Cnn()

>>> my_cnn.set_device('gpu')

>>> result = my_cnn.query('./new_image.jpg', './CNN_Models/model')
>>> print(result)

>>> del my_cnn
```


## Example Images
* Angry

![Angry_exmple](./Train_Images/0_Angry/0.jpg)

* Disgust

![Disgust_exmple](./Train_Images/1_Disgust/0.jpg)

* Fear

![Fear_exmple](./Train_Images/2_Fear/0.jpg)

* Happy

![Happy_exmple](./Train_Images/3_Happy/0.jpg)

* Sad

![Sad_exmple](./Train_Images/4_Sad/0.jpg)

* Surprise

![Surprise_exmple](./Train_Images/5_Surprise/0.jpg)

* Neutral

![Neutral_exmple](./Train_Images/6_Neutral/0.jpg)


## References
* [TensorFlow API r1.4](https://www.tensorflow.org/api_docs/)
