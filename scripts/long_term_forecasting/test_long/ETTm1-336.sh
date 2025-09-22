export CUDA_VISIBLE_DEVICES=0

for lradj in "type1" "type2" "cosine"
do
  for lr in 0.0001 0.00025 0.0005 0.001 0.0025 0.00005
  do
    for batch_size in 8 4
    do
      for stride in 4 8
      do
        python -u long_term_forecast.py \
          --task_name "long_term_forecast" \
          --dataset_name "ETTm1" \
          --data "ETTm1" \
          --root_path "./datasets/ETT/" \
          --data_path "ETTm1.csv" \
          --seq_len 512 \
          --pred_len 336 \
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