export CUDA_VISIBLE_DEVICES=0

for lradj in "type1" "type2" "cosine"
do
  for lr in 0.0001 0.00025
  do
    for batch_size in 16 8 4
    do
      for stride in 4 8
      do
        python -u long_term_forecast.py \
          --task_name "long_term_forecast" \
          --dataset_name "weather" \
          --data "custom" \
          --root_path "./datasets/weather/" \
          --data_path "weather.csv" \
          --seq_len 336 \
          --pred_len 96 \
          --forward_layers 3 \
          --stride $stride \
          --batch_size $batch_size \
          --learning_rate $lr \
          --lradj $lradj \
          --num_workers 5 \
          --enc_in 21 \
          --train_epochs 13 \
          --patience 3 \

        done
      done
    done
done