RUN_NAME=alpaca_pipe0.8
OUTPUT_DIR=/home/LeiFeng/model/json/$RUN_NAME

mkdir -p $OUTPUT_DIR
cp train-alpaca_pipe.sh $OUTPUT_DIR/train.sh
cd ..

CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 --master_port=52000 train-pipe.py \
    --model_name_or_path /home/LeiFeng/json/aliendao/dataroot/models/huggyllama/llama-7b \
    --data_path ./alpaca_rationale.json \
    --bf16 True \
    --output_dir $OUTPUT_DIR \
    --run_name $RUN_NAME \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 16 \
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