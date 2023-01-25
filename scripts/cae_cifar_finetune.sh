OUTPUT_DIR='/opt/data/private/CAE/output/cae-0109-naive-com-FT'
DATA_PATH=/opt/data/private/dataset
TOKENIZER_PATH=/opt/data/private/CAE/tokenizer-weights

ADDRESS=127.0.0.1                                                                                
NNODES=1     
RANK=0                                                                                                                

MODEL_PATH=/opt/data/private/CAE/output/cae-cs-naive-0109/cae-test_checkpoint-59.pth

OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=1 python3 -m torch.distributed.launch \
    --nproc_per_node=1 \
    --nnodes=$NNODES \
    --node_rank=$RANK \
    --master_addr=$ADDRESS \
    --master_port=8899 \
    /opt/data/private/CAE/tools/run_cifar_finetuning.py \
    --model cae_finetune_cifar  --data_path $DATA_PATH \
    --finetune $MODEL_PATH \
    --nb_classes 100 --data_set CIFAR \
    --output_dir $OUTPUT_DIR \
    --batch_size 64 \
    --lr 8e-3 --update_freq 1 \
    --warmup_epochs 5 --epochs 100 --layer_decay 0.65 --drop_path 0.1 \
    --weight_decay 0.05 --mixup 0.8 --cutmix 1.0 \
	--sin_pos_emb \
    --dist_eval \
    --no_auto_resume \
    --exp_name cae-cifar \
    --imagenet_default_mean_and_std

# CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=1 python3 -m torch.distributed.launch --nproc_per_node=2 /opt/data/private/CAE/tools/run_cifar_finetuning.py \
#     --model cae_finetune_cifar  --data_path $DATA_PATH \
#     --finetune $MODEL_PATH \
#     --nb_classes 100 --data_set CIFAR \
#     --output_dir $OUTPUT_DIR \
#     --batch_size 64 \
#     --lr 8e-3 --update_freq 1 \
#     --warmup_epochs 5 --epochs 100 --layer_decay 0.65 --drop_path 0.1 \
#     --weight_decay 0.05 --mixup 0.8 --cutmix 1.0 \
# 	--sin_pos_emb \
#     --dist_eval \
#     --no_auto_resume \
#     --exp_name cae-cifar \
#     --imagenet_default_mean_and_std