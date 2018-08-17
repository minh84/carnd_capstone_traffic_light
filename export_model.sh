#!/bin/bash
TRAIN_TYPE=$1
set -x

python object_detection/export_inference_graph.py \
--pipeline_config_path=config/$TRAIN_TYPE/ssd_mobilenet_v1_coco_carnd.config \
--trained_checkpoint_prefix=finetuned_${TRAIN_TYPE}/ssd_mobilenet_v1_coco/model.ckpt-10000 \
--output_directory=frozen_${TRAIN_TYPE}/ssd_mobilenet_v1_coco

python object_detection/export_inference_graph.py \
--pipeline_config_path=config/$TRAIN_TYPE/ssd_inception_v2_coco_carnd.config \
--trained_checkpoint_prefix=finetuned_${TRAIN_TYPE}/ssd_inception_v2_coco/model.ckpt-10000 \
--output_directory=frozen_${TRAIN_TYPE}/ssd_inception_v2_coco

python object_detection/export_inference_graph.py \
--pipeline_config_path=config/$TRAIN_TYPE/faster_rcnn_resnet101_coco_carnd.config \
--trained_checkpoint_prefix=finetuned_${TRAIN_TYPE}/faster_rcnn_resnet101_coco/model.ckpt-10000 \
--output_directory=frozen_${TRAIN_TYPE}/faster_rcnn_resnet101_coco