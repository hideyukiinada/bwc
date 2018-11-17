# Generate a color portrait from a grayscale portrait
The goal of this project is to generate a color photo of a human face from a grayscale portrait image using an autoencoder.

## Status of the project
This project is still in the research phase.

### Issues that need to be resolved
#### Externally-visible issues
- Generated faces are distorted
- Details are lost
#### Loss
- Though loss for the training set steadily decreases during training, validation loss starts to increase around after 1200 epochs, indicating overfitting.  Color images generated at 1200 epochs still suffer from the visible issues above, so I need to figure out a way to address these issues.  

## Dataset used for training and testing
I am using UMass [Labeled Faces in the Wild](http://vis-www.cs.umass.edu/lfw/).  Files were resized to 256x256-pixel images.  For input, files were converted to single-channel grayscale images then fed to the script.  The objective of training is to minimize the MSE between a generated color image and an original 256x256-pixel color image.
