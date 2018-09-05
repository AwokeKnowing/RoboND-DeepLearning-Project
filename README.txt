The main task in this project is to process virtual webcam images from a virtual quadcoptor and output an image that labels each pixel as either 'hero' (person to follow), 'pedestrian', or 'other' by ouputing an image the same size as the input image but with each pixel replaced with one of 3 colors corresponding to the class (hero, pedestrian, other).

### Environment
The model is coded and trained using Keras/Tensorflow (tested on Tensorflow 1.2.1 and 1.10.0). The network runs in real time and a quadcopter simulator (RoboND-QuadRotor-Unity-Simulator by Udacity) feeds synthetic webcam images to the trained model and uses the output image to locate the 'hero' and follow them at a distance.  
The problem of actually following a person using a segmented image is not addressed in this project, but the method described here is validated using the simulator which successfully follows the 'hero' given the stream of segmented images processed by the network described in this project.

### Solution
To complete the task, we preprocess the data, build the model, choose hyperparameters, train the network, and validate the network using a held out set of data.  The held out set of data is split into sets of data emphazising separate classes or conditions (such as 'hero' is visible but far away), to simplify measuring how well the model performs at each of the tasks.
The main code for the solution is implemented in a Jupyter notebook (`code/model_training.ipynb`).
#### Preprocessing
The Udacity simulator is used to generate a dataset of synthetic images. Images are downsampled to 160x160, and training data is labeled via masks created by information provided by the simulator.
#### Fully Convulutional Network Model
The model consists of 4 depth-wise separable 'same' convolutional layers (stride=2, kernel size=3, with batchnorm), followed by a 1x1 convolution and then 4 deconvolutional layers.  Each of the deconvolutional layers receives the concatenated input of the same size from the convolutional side.  The deconvolutional layers use bilinear upscaling before applying 2 depth-wise separable 'same' convulutions. A final convolutional layer with 3 filters and softmax activation is used to generate the final prediction masks for each of the 3 classes ('hero', pedestrian, other).

#### Results
Using the IOU metric, the trained model achieved a modest 40% accuracy.
Much of the classification error was due to the weighting of false negative 'hero' detections.  When the 'hero' is just a few pixels and is missed, more 'accuracy' is lost than when the same number of pixels are misclassified on a larger object instances.
The following hyperparameters were used in training the network.  Comprehensive parameter search was not performed.  Values are standard in the literature, excepts for epochs which is fairly low for CNN image training, but sufficient.
| parameter | value | comment |
| learning rate | .002 | .001 (common starting point) was unstable |
| batch size | 16 | classic low batch size as a regulator |
| epochs | 20 | fairly low to avoid overfitting |
| validation steps | 1 | validation-based parameter search was not done, so not useful |
| samples | 4096 | 4131 images in provided data |
| encoding blocks | 4 | downsampling to where whole-scene concepts could emerge |
| decoding blocks | 4 | upsampling at same rate to facilitate skip connections |
| filters | 2 ^ (l+4) | keep overal info content by trading filter size for depth
at .002 for 20 epochs 36%
#### Future 
There are several directions for future research which would likely yeild significant benefits in training speed and accuracy.  
One direction would be to use the full resolution images to improve the detection of the 'hero' when they are far away (only a few pixels).  Challenges would include finding efficient ways to to downsample and upsample the images without needed to add too many additional layers.
Another direction would be to quantify the best case scenario for the current model given several orders of magnatude more data.  While training time would increase, inference using the resulting model would run at the same speed, so any improvements in accuracy would presumably be well worth the training.
Additionally, the model itself could be rearchitectured to use skip connections more sparingly, and regularization/dropout and other common layers could assist in improving the model.  LSTM layers could potentially help stabilize the model for the hero at far distances by 'remembering'/expecting the 'hero' to be in a certain location .   And attention could be used to make the skip connections more focused on areas of interest.
