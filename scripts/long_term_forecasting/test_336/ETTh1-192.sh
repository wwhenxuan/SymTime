export CUDA_VISIBLE_DEVICES=4

for lradj in "type1" "type2" "cosine"
do
  for lr in 0.0001 0.00025 0.00075 0.001
  do
    for batch_size in 8 4 16 32
    do
      for stride in 4 8
      do
        python -u long_term_forecast.py \
          --task_name "long_term_forecast" \
          --dataset_name "ETTh1" \
          --data "ETTh1" \
          --root_path "./datasets/ETT/" \
          --data_path "ETTh1.csv" \
          --seq_len 336 \
          --pred_len 192 \
          --forward_layers 3 \
          --stride $stride \
          --batch_size $batch_size \
          --learning_rate $lr \
          --lradj $lradj \
          --num_workers 5 \
          --enc_in 7 \
          --train_epochs 16 \
          --patience 3 \

        done
      done
    done
done