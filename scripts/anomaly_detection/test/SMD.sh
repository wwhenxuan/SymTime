export CUDA_VISIBLE_DEVICES=5

# try forward_layers = 2
# already layers in 3, batch_size in 32 16

for batch_size in 64 32 16
do
  for forward_layers in 2 3
  do
    for stride in 4 8
    do
      for lradj in "type1" "type2" "cosine"
      do
        for lr in 0.0001 0.00025 0.0005 0.00075
        do
          python -u anomaly_detection.py \
            --task_name "anomaly_detection" \
            --dataset_name "SMD" \
            --data "SMD" \
            --root_path "./datasets/SMD" \
            --anomaly_ratio 0.5 \
            --forward_layers $forward_layers \
            --stride $stride \
            --batch_size $batch_size \
            --learning_rate $lr \
            --lradj $lradj \
            --enc_in 38 \
            --train_epochs 12 \
            --seq_len 100 \

        done
      done
    done
  done
done
