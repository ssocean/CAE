# tmp_my_name=${0##*/}
# my_name=${tmp_my_name%.*}
# my_name = cae-test
OUTPUT_DIR='/opt/data/private/CAE/output/cae-decoupled-0120'
DATA_PATH=/opt/data/private/dataset
TOKENIZER_PATH=/opt/data/private/CAE/tokenizer-weights
ADDRESS=127.0.0.1                                                                                
NNODES=1     
RANK=0                                                                                                                        

CUDA_VISIBLE_DEVICES=0,1 python3 -m torch.distributed.launch --nproc_per_node=2 /opt/data/private/CAE/tools/pretraining_decoupled.py \
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

