#!/bin/bash
set -x

python object_detection/export_inference_graph.py \
--pipeline_config_path=config/sim/ssd_mobilenet_v1_coco_carnd_sim.config \
--trained_checkpoint_prefix=finetuned_sim/ssd_mobilenet_v1_coco/model.ckpt-10000 \
--output_directory=frozen_sim/ssd_mobilenet_v1_coco

python object_detection/export_inference_graph.py \
--pipeline_config_path=config/sim/ssd_inception_v2_coco_carnd_sim.config \
--trained_checkpoint_prefix=finetuned_sim/ssd_inception_v2_coco/model.ckpt-10000 \
--output_directory=frozen_sim/ssd_inception_v2_coco

python object_detection/export_inference_graph.py \
--pipeline_config_path=config/sim/faster_rcnn_resnet101_coco_carnd_sim.config \
--trained_checkpoint_prefix=finetuned_sim/faster_rcnn_resnet101_coco/model.ckpt-10000 \
--output_directory=frozen_sim/faster_rcnn_resnet101_coco