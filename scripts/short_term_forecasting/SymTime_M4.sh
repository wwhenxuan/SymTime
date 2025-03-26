export CUDA_VISIBLE_DEVICES=0

# Yearly SMAPE: 13.366, MASE: 2.999, OWA: 0.786
python -u fine_tuning.py \
  --task_name "short_term_forecast" \
  --dataset_name "m4Benchmark" \
  --model "Model" \
  --root_path "./datasets/m4/" \
  --data_path "m4" \
  --seasonal_patterns "Yearly" \
  --batch_size 8 \
  --lradj "type2" \
  --learning_rate 0.0002 \
  --forward_layers 3 \
  --stride 4 \
  --seed 2025 \

# Monthly SMAPE: 12.608, MASE: 0.925, OWA: 0.872
python -u fine_tuning.py \
  --task_name "short_term_forecast" \
  --dataset_name "m4Benchmark" \
  --model "Model" \
  --root_path "./datasets/m4/" \
  --data_path "m4" \
  --seasonal_patterns "Monthly" \
  --batch_size 32 \
  --lradj "cosine" \
  --learning_rate 0.0001 \
  --forward_layers 3 \
  --stride 1 \
  --seed 2025 \

# Quarterly
python -u fine_tuning.py \
  --task_name "short_term_forecast" \
  --dataset_name "m4Benchmark" \
  --model "Model" \
  --root_path "./datasets/m4/" \
  --data_path "m4" \
  --seasonal_patterns "Quarterly" \
  --batch_size 32 \
  --lradj "cosine" \
  --learning_rate 0.0001 \
  --forward_layers 3 \
  --stride 1 \
  --seed 2025 \

# Others SMAPE: 4.941 MASE: 3.327 OWA: 1.045
python -u fine_tuning.py \
  --task_name "short_term_forecast" \
  --dataset_name "m4Benchmark" \
  --model "Model" \
  --root_path "./datasets/m4/" \
  --data_path "m4" \
  --seasonal_patterns "Daily" \
  --batch_size 32 \
  --lradj "cosine" \
  --learning_rate 0.0002 \
  --forward_layers 3 \
  --stride 1 \
  --seed 2025 \

python -u fine_tuning.py \
  --task_name "short_term_forecast" \
  --dataset_name "m4Benchmark" \
  --model "Model" \
  --root_path "./datasets/m4/" \
  --data_path "m4" \
  --seasonal_patterns "Weekly" \
  --batch_size 32 \
  --lradj "cosine" \
  --learning_rate 0.0002 \
  --forward_layers 3 \
  --stride 1 \
  --seed 2025 \

python -u fine_tuning.py \
  --task_name "short_term_forecast" \
  --dataset_name "m4Benchmark" \
  --model "Model" \
  --root_path "./datasets/m4/" \
  --data_path "m4" \
  --seasonal_patterns "Hourly" \
  --batch_size 32 \
  --lradj "cosine" \
  --learning_rate 0.0002 \
  --forward_layers 3 \
  --stride 1 \
  --seed 2025 \



