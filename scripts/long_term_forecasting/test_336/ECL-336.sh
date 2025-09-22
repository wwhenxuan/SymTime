export CUDA_VISIBLE_DEVICES=5

for lradj in "type1" "type2" "cosine"
do
  for lr in 0.0001 0.00025 0.00075 0.001
  do
    for batch_size in 8 4
    do
      for stride in 8
      do
        python -u long_term_forecast.py \
          --task_name "long_term_forecast" \
          --dataset_name "ECL" \
          --data "custom" \
          --root_path "./datasets/electricity/" \
          --data_path "electricity.csv" \
          --seq_len 336 \
          --pred_len 336 \
          --forward_layers 3 \
          --stride $stride \
          --batch_size $batch_size \
          --learning_rate $lr \
          --lradj $lradj \
          --num_workers 5 \
          --enc_in 321 \
          --train_epochs 16 \
          --patience 3 \

        done
      done
    done
done