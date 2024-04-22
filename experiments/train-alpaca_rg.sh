RUN_NAME=alpaca_rg-rationale_srkld_loss0.7
OUTPUT_DIR=/data/data8/models/json/$RUN_NAME

mkdir -p $OUTPUT_DIR
cp train-alpaca_rg.sh $OUTPUT_DIR/train.sh
cd ..

CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 --master_port=52002 train-rg.py \
    --model_name_or_path /home/LeiFeng/json/aliendao/dataroot/models/huggyllama/llama-7b \
    --data_path ./alpaca_rationale.json \
    --bf16 True \
    --output_dir $OUTPUT_DIR \
    --run_name $RUN_NAME \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 32 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
    --tf32 True \
| tee -a $OUTPUT_DIR/train.log