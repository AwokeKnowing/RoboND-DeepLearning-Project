# RoboND-DeepLearning-Project
Fully Convolutional Net for semantic segmentation of Quadcoptor webcam stream

The main task in this project is to process virtual webcam images from a virtual quadcoptor and output an image that labels each pixel as either 'hero' (person to follow), 'pedestrian', or 'other' by ouputing an image the same size as the input image but with each pixel replaced with one of 3 colors corresponding to the class (hero, pedestrian, other).

### Environment
The model is coded and trained using Keras/Tensorflow. The network runs in real time and a quadcopter simulator (RoboND-QuadRotor-Unity-Simulator by Udacity) feeds synthetic webcam images to the trained model and uses the output image to locate the 'hero' and follow them at a distance.  
The problem of actually following a person using a segmented image is not addressed in this project, but the method described here is validated using the simulator which successfully follows the 'hero' given the stream of segmented images processed by the network described in this project.

### Solution
To complete the task, we preprocess the data, build the model, choose hyperparameters, train the network, and validate the network using a held out set of data.  The held out set of data is split into sets of data emphazising separate classes or conditions (such as 'hero' is visible but far away), to simplify measuring how well the model performs at each of the tasks.
The main code for the solution is implemented in a Jupyter notebook (`code/model_training.ipynb`).
#### Preprocessing
The Udacity simulator is used to generate a dataset of synthetic images. Images are downsampled to 160x160, and training data is labeled via masks created by information provided by the simulator.
#### Fully Convulutional Network Model
The model consists of 3 depth-wise separable 'same' convolutional layers (stride=2, kernel size=3, with batchnorm), followed by a 1x1 convolution and then 3 deconvolutional layers.  Each of the deconvolutional layers receives the concatenated input of the same size from the convolutional side.  The deconvolutional layers use bilinear upscaling before applying 2 depth-wise separable 'same' convulutions. A final convolutional layer with 3 filters and softmax activation is used to generate the final prediction masks for each of the 3 classes ('hero', pedestrian, other).

### Results
Using the IoU metric, the trained model achieved a modest 41% accuracy. The IoU metric measures the proportion of correctly labeled pixels to total pixels labeled.
Much of the classification error was due to the weighting of false negative 'hero' detections.  The issue is most likely an issue with the training distribution which had relatively few examples of the 'hero' vs other pedestrians.
The following hyperparameters were used in training the network.  Comprehensive parameter search was not performed, but approximately 20 different combinations of parameters were attempted before arriving at the best score reported here (41% average IoU for labeling pedestrians and the 'hero').  Values are in standard ranges found in the literature. In fact after trying many combinations of learning rate, batches and batch size and epochs, in the end the best choice was the classic batch size 16 with learning rate 1e-3. The result was obtained by simply training for more epochs (120)
| parameter | value | comment |
| learning rate | .001 | also tried .00001, .01, .005|
| batch size | 16 | classic low batch size as a regulator |
| epochs | 120 | random mini-batches |
| validation steps | 8 | validation-based parameter search was not done |
| samples | 4096 | 4131 images in provided data |
| encoding blocks | 3 | downsampling to where whole-scene concepts could emerge |
| decoding blocks | 3 | upsampling at same rate to facilitate skip connections |
| filters | 32->256->32 | keep overal info content by trading filter size for depth

### Future 
There are several directions for future research which would likely yeild significant benefits in training speed and accuracy.  
One direction would be to use the full resolution images to improve the detection of the 'hero' when they are far away (only a few pixels).  Challenges would include finding efficient ways to to downsample and upsample the images without needed to add too many additional layers.
Another direction would be to quantify the best case scenario for the current model given several orders of magnatude more data.  While training time would increase, inference using the resulting model would run at the same speed, so any improvements in accuracy would presumably be well worth the training.
Additionally, the model itself could be rearchitectured to use skip connections more sparingly, and regularization/dropout and other common layers could assist in improving the model.  LSTM layers could potentially help stabilize the model for the hero at far distances by 'remembering'/expecting the 'hero' to be in a certain location .   And attention could be used to make the skip connections more focused on areas of interest.
