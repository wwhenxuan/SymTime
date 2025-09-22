export CUDA_VISIBLE_DEVICES=

for batch_size in 16
do
  for learning_rate in 0.0002
  do
    python -u short_term_forecast.py \
      --task_name "short_term_forecast" \
      --dataset_name "m4Benchmark" \
      --model "Model6" \
      --root_path "./datasets/m4/" \
      --data_path "m4" \
      --seasonal_patterns "Yearly" \
      --batch_size $batch_size \
      --lradj "type1" \
      --learning_rate $learning_rate \
      --forward_layers 3 \
      --stride 4 \
      --seed 2025 \

    # Monthly SMAPE: 12.608, MASE: 0.925, OWA: 0.872
    python -u short_term_forecast.py \
      --task_name "short_term_forecast" \
      --dataset_name "m4Benchmark" \
      --model "Model6" \
      --root_path "./datasets/m4/" \
      --data_path "m4" \
      --seasonal_patterns "Monthly" \
      --batch_size $batch_size \
      --lradj "type1" \
      --learning_rate $learning_rate \
      --forward_layers 3 \
      --stride 1 \
      --seed 2025 \

    # Quarterly
    python -u short_term_forecast.py \
      --task_name "short_term_forecast" \
      --dataset_name "m4Benchmark" \
      --model "Model6" \
      --root_path "./datasets/m4/" \
      --data_path "m4" \
      --seasonal_patterns "Quarterly" \
      --batch_size $batch_size \
      --lradj "type1" \
      --learning_rate $learning_rate \
      --forward_layers 3 \
      --stride 1 \
      --seed 2025 \

    # Others SMAPE: 4.941 MASE: 3.327 OWA: 1.045
    python -u short_term_forecast.py \
      --task_name "short_term_forecast" \
      --dataset_name "m4Benchmark" \
      --model "Model6" \
      --root_path "./datasets/m4/" \
      --data_path "m4" \
      --seasonal_patterns "Daily" \
      --batch_size $batch_size \
      --lradj "cosine" \
      --learning_rate $learning_rate \
      --forward_layers 3 \
      --stride 1 \
      --seed 2025 \

    python -u short_term_forecast.py \
      --task_name "short_term_forecast" \
      --dataset_name "m4Benchmark" \
      --model "Model2" \
      --root_path "./datasets/m4/" \
      --data_path "m4" \
      --seasonal_patterns "Weekly" \
      --batch_size $batch_size \
      --lradj "cosine" \
      --learning_rate $learning_rate \
      --forward_layers 3 \
      --stride 1 \
      --seed 2025 \

    python -u short_term_forecast.py \
      --task_name "short_term_forecast" \
      --dataset_name "m4Benchmark" \
      --model "Model6" \
      --root_path "./datasets/m4/" \
      --data_path "m4" \
      --seasonal_patterns "Hourly" \
      --batch_size $batch_size \
      --lradj "cosine" \
      --learning_rate $learning_rate \
      --forward_layers 3 \
      --stride 1 \
      --seed 2025 \

  done
done

