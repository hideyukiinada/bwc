# Generate a color portrait from a grayscale portrait
The goal of this project is to generate a color photo of a human face from a grayscale portrait image using an autoencoder.

## Status of the project
This project is still in the research phase.

## Dataset used for training and testing
I am using UMass [Labeled Faces in the Wild!](http://vis-www.cs.umass.edu/lfw/).  Files were resized by 256x256-pixel image.  For input, files were converted to single-channel grayscale images then fed to the script.  The objective of training is to minimize the MSE between a generated color image and an original color image.
