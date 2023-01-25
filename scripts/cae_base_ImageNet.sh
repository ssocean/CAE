tmp_my_name=${0##*/}
my_name=${tmp_my_name%.*}

OUTPUT_DIR='./output/'$my_name
DATA_PATH=/path/to/imagenet1k/train
TOKENIZER_PATH=./tokenizer-weights

ADDRESS=127.0.0.1                                                                             
NNODES=1   
RANK=0                                                                                                                    

# ============================ pretraining ============================
OMP_NUM_THREADS=1 python3 -m torch.distributed.launch \
  --nproc_per_node=8 \
  --nnodes=$NNODES \
  --node_rank=$RANK \
  --master_addr=$ADDRESS \
  --master_port=8899 \
  /opt/data/private/CAE/tools/run_pretraining.py \
  --data_path ${DATA_PATH} \
  --output_dir ${OUTPUT_DIR} \
  --model cae_base_patch16_224_8k_vocab --discrete_vae_weight_path ${TOKENIZER_PATH} \
  --batch_size 64 --lr 1.5e-3 --warmup_epochs 20 --epochs 800 \
  --clip_grad 3.0 --layer_scale_init_value 0.1 \
  --imagenet_default_mean_and_std \
  --color_jitter 0 \
  --drop_path 0.1 \
  --sincos_pos_emb \
  --mask_generator block \
  --num_mask_patches 98 \
  --decoder_layer_scale_init_value 0.1 \
  --no_auto_resume \
  --save_ckpt_freq 100 \
  --exp_name $my_name \
  --regressor_depth 4 \
  --decoder_depth 4 \
  --align_loss_weight 2

