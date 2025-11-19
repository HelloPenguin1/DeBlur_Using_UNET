Image Deblurring Application
Project Overview
This project implements an end-to-end pipeline for restoring blurred images using a deep learning approach (U-Net architecture) with TensorFlow/Keras. Each file in the codebase is structured for clarity, modularity, and ease of extension.

Component Breakdown
1. config.py
Role: Centralizes all configuration parameters and important paths.

Key Variables:

IMGWIDTH, IMGHEIGHT: Target resolution (256x256) for all input and output images, ensuring a fixed input size for the model.

SCRIPTDIR, ROOTDIR: Paths to script and root directories, making file path management flexible and robust.

MODELFILENAME, MODELPATH: Points to the U-Net model weights; easy to switch models by changing just this line.

INPUTIMAGEPATH: Path to the sample blurred image to be deblurred; modify here to try other images.

2. utils.py
Role: Utility functions for image pre-processing and visualization.

Main Functions:

loadimage(path):

Reads the image from disk, converts from BGR to RGB (OpenCV default), resizes to 256x256, and normalizes pixel values to.​

Useful for standardizing image inputs regardless of original format or size.

showresults(original, deblurred):

Displays the original (blurred) and deblurred images side-by-side using matplotlib.

Aids in visual performance assessment of the model, making results easily interpretable.

3. unet_archi.py
Role: Defines the U-Net model architecture and all the inner building blocks.

Contents:

Encoder Block:

Two convolutional layers (Conv2D) with batch normalization and ReLU activations, followed by max pooling.

Purpose: Extract hierarchical spatial features and progressively downsample the input.

Decoder Block:

Upsampling, concatenates with skip connections from the encoder, followed by convolutions and activations.

Purpose: Gradually reconstruct the image, integrating high-resolution features from earlier layers via skips.

builddeblurringcnn():

Assembles the U-Net using encoder and decoder blocks.

Adds input to output for residual learning, helping the model focus on correcting the blur.

Final output is clipped between 0 and 1 for clean image data, ensuring pixel values remain valid.

4. model_loader.py
Role: Encapsulates model loading and inference processes.

Main Functions:

loadmodel(path):

Creates the U-Net using the architecture, and loads pre-trained weights from an H5 file.

Robust error handling for missing or corrupt weight files.

predict(model, img):

Adds a batch dimension to a single image (as models expect batches), runs inference with the network, and post-processes the output (removes batch dimension, clips values to ).​

Produces the deblurred image ready for display.

5. main.py
Role: Provides the executable workflow (main program logic).

Detailed Workflow:

Prints title banner for clarity.

Loads input image:

Checks for existence of the image file and displays an error if not found.

Uses utils.loadimage.

Loads the model:

Loads weights and architecture, verifies the file exists.

Uses model_loader.loadmodel.

Deblurring process:

Runs the input image through the model via model_loader.predict.

Displays results:

Uses utils.showresults to show a side-by-side visual comparison of input and output.

Handles all errors gracefully with informative printouts.

How to Use
Update configuration in config.py if you want to use different images or models.

Run main.py to start the deblurring pipeline. Follow on-screen prompts for status and error messages. The results will be displayed graphically at the end.

