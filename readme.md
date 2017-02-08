# brief

## Model

The model is defined in the model.py. It basically consists of one vgg to encode the image, and one very simple rnn to encode the sentence.

## Training and Testing

Run the tool/train_net.py to train the network. 
And run the tool/test_retrieval.py to do the retrieval.
In the beginning part of the train_net.py and test_retrieval.py,
many flags are defined (just like a ordinary tensorflow project). 
Besides that I used an easydict to store the rest of the flags,
most of which will not be changed frequently.
They are stored in the config.py.

Note that in order to use pretrained vgg model, I wrote a special network loader interface.
But it seems that it is less convienient than I had thought.
I hope to switch back to the standard tensorflow restorer soon.

And it will be a good idea to combine the testing and training into one python file.

## data

Most of the data are stored in /ais/gobi4/tingwuwang/joint_embedding.
I am currently using the bird dataset (8000 images with 5 sentences each I think?).
But coco, flicker etc. should be coming later.

The data interface is written in the util/util.py

# preliminary

## Tensorflow > 0.9

## python package

easydict (pip install)

torchfile (pip install)

scikit-image (pip install)
