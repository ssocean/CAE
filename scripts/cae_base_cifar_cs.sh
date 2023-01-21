# tmp_my_name=${0##*/}
# my_name=${tmp_my_name%.*}
# my_name = cae-test
OUTPUT_DIR='/opt/data/private/CAE/output/cae-cs-0111'
DATA_PATH=/opt/data/private/dataset
TOKENIZER_PATH=/opt/data/private/CAE/tokenizer-weights
ADDRESS=127.0.0.1                                                                                
NNODES=1     
RANK=0                                                                                                                        
# ============================ pretraining ============================
# OMP_NUM_THREADS=1 python3 -m torch.distributed.launch \
#   --nproc_per_node=1 \
#   --nnodes=$NNODES \
#   --node_rank=$RANK \
#   --master_addr=$ADDRESS \
#   --master_port=8899 \

  # OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 /opt/data/private/CAE/tools/run_pretraining_cs.py \
  # --data_path ${DATA_PATH} \
  # --output_dir ${OUTPUT_DIR} \
  # --model com_cae_cifar --discrete_vae_weight_path ${TOKENIZER_PATH} \
  # --batch_size 32 --lr 1.5e-3 --warmup_epochs 20 --epochs 800 \
  # --clip_grad 3.0 --layer_scale_init_value 0.1 \
  # --imagenet_default_mean_and_std \
  # --color_jitter 0 \
  # --drop_path 0.1 \
  # --sincos_pos_emb \
  # --ratio_mask_patches 0.25 \
  # --mask_generator complementary \
  # --num_mask_patches 16 \
  # --decoder_layer_scale_init_value 0.1 \
  # --no_auto_resume \
  # --save_ckpt_freq 100 \
  # --exp_name cae-test \
  # --regressor_depth 4 \
  # --decoder_depth 4 \
  # --align_loss_weight 2 \
  # --num_workers 0
  # --warmup_epochs 20 --epochs 400 \ 
  # 100
CUDA_VISIBLE_DEVICES=0,1 python3 -m torch.distributed.launch --nproc_per_node=2 /opt/data/private/CAE/tools/run_pretraining_cs.py \
  --data_path ${DATA_PATH} \
  --output_dir ${OUTPUT_DIR} \
  --model com_cae_cifar --discrete_vae_weight_path ${TOKENIZER_PATH} \
  --batch_size 32 --lr 1.5e-3 --warmup_epochs 20 --epochs 400 \
  --clip_grad 3.0 --layer_scale_init_value 0.1 \
  --imagenet_default_mean_and_std \
  --color_jitter 0 \
  --drop_path 0.1 \
  --sincos_pos_emb \
  --ratio_mask_patches 0.25 \
  --mask_generator complementary \
  --num_mask_patches 16 \
  --decoder_layer_scale_init_value 0.1 \
  --no_auto_resume \
  --save_ckpt_freq 5 \
  --exp_name cae-test \
  --regressor_depth 4 \
  --decoder_depth 4 \
  --align_loss_weight 2 \
  --num_workers 0

# ============================ linear probing ============================
# DATA_PATH=/path/to/imagenet1k/
# MODEL_PATH=/path/to/pretrained/model

# OMP_NUM_THREADS=1 python -m torch.distributed.launch \
#     --nproc_per_node=8 \
#     --nnodes=$NNODES \
#     --node_rank=$RANK \
#     --master_addr=$ADDRESS \
#     --master_port=8899 \
#     tools/run_linear.py \
#     --model cae_base_patch16_224 --data_path $DATA_PATH \
#     --finetune $MODEL_PATH \
#     --nb_classes 1000 \
#     --batch_size 512 \
#     --epochs 90 \
#     --blr 0.1 \
#     --weight_decay 0.0 \
#     --dist_eval --data_path ${DATA_PATH} \
#     --output_dir $OUTPUT_DIR \
#     --log_dir $OUTPUT_DIR \
#     --enable_linear_eval \
#     --use_cls \
#     --dist_eval \
#     --save_freq 50 \
#     --disable_rel_pos_bias \
#     --linear_type standard \
#     --exp_name $my_name

# # ============================ attentive probing ============================
# DATA_PATH=/path/to/imagenet1k/
# MODEL_PATH=/path/to/pretrained/model

# OMP_NUM_THREADS=1 python -m torch.distributed.launch \
#     --nproc_per_node=8 \
#     --nnodes=$NNODES \
#     --node_rank=$RANK \
#     --master_addr=$ADDRESS \
#     --master_port=8899 \
#     tools/run_attentive.py \
#     --model cae_base_patch16_224 --data_path $DATA_PATH \
#     --finetune $MODEL_PATH \
#     --nb_classes 1000 --data_set IMNET --imagenet_default_mean_and_std \
#     --output_dir $OUTPUT_DIR --batch_size 256 --lr 0.4 --update_freq 1 \
#     --warmup_epochs 10 --epochs 90 \
#     --weight_decay 0 --smoothing 0.0 --layer_decay 1.0 --drop_path 0.0 \
#     --color_jitter 0.0 --mixup 0.0 --cutmix 0.0 --reprob 0.0 \
#     --opt sgd --momentum 0.9 \
#     --enable_linear_eval \
#     --use_cls \
#     --dist_eval \
#     --no_auto_resume \
#     --save_ckpt_freq 50 \
#     --linear_type attentive \
#     --exp_name $my_name
