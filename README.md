# Final Project - Artistic Style Transfer (ECBM4040)

The project is a recreation of the research paper [A Neural Algorithm of Artistic Style](https://arxiv.org/pdf/1508.06576.pdf).



## Directory Structure
- ./Art.ipynb - The main python notebook containing code to run the file and generate artistic images. It takes input from the data, model and helper_funcs directories to create the 

```
Art
│   README.md
│   Art.ipynb #main jupyter notebook    
│
└───data
│   │
│   └───content_images #directory for content images
|   │   │   content_2.jpg 
│   │   │   content_3.jpg
|   |   │   ...
│   │
│   └───output #directory for output images
│   │   │   outfile.jpg
│   │   │   ...
│   │
│   └───style_images #directory for style images
│   │   │   style.jpg
│   │   │   style1.jpg
│   │   │   ...
│   
└───helper_funcs #helper functions for project
│   │   __init__.py #file such that other python files can be imported
│   │   helper.py #helper functions which read and write images, find losses and train models
│   │   vgg.py #builds vgg model
│
└───model #model directory
│   │
│   └───vgg_19 #vgg 19
│   │   │   imagenet-vgg-verydeep-19.mat #matrix file for vgg
```

## Instructions

The mat file for the pre-trained VGG model are too large.
It is required that you download them from the below mentioned website, and place it in the path model/vgg_model


[VGG-19](http://www.vlfeat.org/matconvnet/models/beta16/imagenet-vgg-verydeep-19.mat)


## Usage
The main jupyter notebook is art.ipynb.

Required to download VGG weights.(As given in instructions)


To change the content image, upload the selected image to data/content_images and change the path in art.ipynb

To change the cstyle image, upload the selected image to data/style_images and change the path in art.ipynb

All the code blocks of the notebook must be run in order to get the output. 


## License
[MIT](https://choosealicense.com/licenses/mit/)