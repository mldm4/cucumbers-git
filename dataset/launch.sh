PATH_TO_YOUR_PIPELINE_CONFIG=
PATH_TO_TRAIN_DIR=
PATH_TO_EVAL_DIR=

python object_detection/python object_detection/train.py \
    --logtostderr \
    --pipeline_config_path=${PATH_TO_YOUR_PIPELINE_CONFIG} \
    --train_dir=${PATH_TO_TRAIN_DIR}

python object_detection/eval.py \
    --logtostderr \
    --pipeline_config_path=${PATH_TO_YOUR_PIPELINE_CONFIG} \
    --checkpoint_dir=${PATH_TO_TRAIN_DIR} \
    --eval_dir=${PATH_TO_EVAL_DIR}

#python object_detection/export_inference_graph.py \
#    --input_type image_tensor \
#    --pipeline_config_path ${PIPELINE_CONFIG_PATH} \
#    --trained_checkpoint_prefix ${TRAIN_PATH} \
#    --output_directory ${EXPORT_DIR}

python -m object_detection/inference/infer_detections \
  --input_tfrecord_paths=$TF_RECORD_FILES \
  --output_tfrecord_path=${SPLIT}_detections.tfrecord-00000-of-00001 \
  --inference_graph=faster_rcnn_inception_resnet_v2_atrous_oid/frozen_inference_graph.pb \
  --discard_image_pixels

python -m object_detection/metrics/offline_eval_map_corloc \
  --eval_dir=${SPLIT}_eval_metrics \
  --eval_config_path=${SPLIT}_eval_metrics/${SPLIT}_eval_config.pbtxt \
  --input_config_path=${SPLIT}_eval_metrics/${SPLIT}_input_config.pbtxt
