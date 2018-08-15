# Traffic Light Detection Export Inference Model
This repository show how we export trained model to be used in the Carnd Capstone Project. The reason we setup this repository is due to

* Udacity only support Tensorflow 1.3.0
* Tensorflow 1.3.0 is too outdated (a year ago) which doesn't have new improvement in Object Detection

Following a suggestion from our [Slack](, we decide to adapt an ad-hoc solution:
 
* we train model using Tensorflow-GPU 1.8.0 + Python 3.6: this work is done in this [repository]())
* we export model using Tensorflow 1.3.0 + Python 2.7: this is described in the following.

## Setup
The step consists the following steps:
* install a Python environment, we use [Miniconda](https://conda.io/miniconda.html)
```bash
conda create -n py2_tf1_3 python=2
conda activate py2_tf1_3
```
then clone this repository and go into it root directory.

* install required packages with the following command
```bash
pip install -r requirements.txt 
pip install ipython jupyter notebook
``` 
we also need to update the `protoc` to later version
```bash
conda install protoc==3.6.0
```

* we have already setup the object detection modules (in subdirectories `object_detection` and `slim`) in our repository, here is the detail steps of how we set up it so that it's compatible (partially) with Tensorflow 1.3  

    * get the [Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection) by running
    ```bash
    git clone https://github.com/tensorflow/models.git
    ```
    
    * however we can't use master since it is not compatible with Tensorflow 1.3.0. We need to checkout the following version
    ```bash
    git checkout d135ed9c04bc9c60ea58f493559e60bc7673beb7
    ```
    this is the latest commit before they include `tf.keras` in the [Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection). 
    
    * Then for our usage, we keep only `object_detection` and `slim`
    ```bash
    cp -r models/research/object_detection .
    cp -r models/research/slim .
    rm -r models
    ```

* note that the code for Faster-RCNN (see `faster_rcnn_meta_arch.py`) uses `tf.AUTO_REUSE` (introduced in Tensorflow 1.4.0), so we need do a patch to make it compatible with Tensorflow 1.3.0.
```bash
cp -r patch_for_tf1_3/* object_detection/
```

* next step is to compile protoc file (ref [here](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md))
```bash
protoc object_detection/protos/*.proto --python_out=.
``` 
 
## Export Inference Model
After trained model in this [repository](), one can export trained model using the following command
```bash
python object_detection/legacy/train.py \
--pipeline_config_path=config/sim/ssd_mobilenet_v1_coco_carnd_sim.config \
--train_dir=finetuned_sim/ssd_mobilenet_v1_coco
```
This command will export our trained model into a single file `frozen_inference_graph.pb` inside the specified `train_dir`. 

## Test Inference Model 
We consider the three following models  

* ssd_mobilenet_v1_coco
* ssd_inception_v2_coco
* faster_rcnn_resnet101_coco

We measure the following metrics for both simulator and real data

* detection accuracy 
* inference time 

The analysis is done in the notebook `notebooks`, we obtain the following result

