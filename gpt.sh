MODEL=gpt-4
KEY=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

python -u gpt.py \
--model $MODEL \
--data_path ./alpaca_data.json \
--output_path ./alpaca_rationale.json \
--key $KEY \
| tee -a generate.log