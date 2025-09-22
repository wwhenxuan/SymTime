export CUDA_VISIBLE_DEVICES=7

for batch_size in 16 32
do
  for forward_layers in 3
  do
    for stride in 8
    do
      for lradj in "type1" "type2" "cosine"
      do
        for lr in 0.0001 0.00025 0.0005 0.00075
        do
          python -u anomaly_detection.py \
            --task_name "anomaly_detection" \
            --dataset_name "SWaT" \
            --data "SWAT" \
            --root_path "./datasets/SWaT" \
            --anomaly_ratio 1 \
            --forward_layers $forward_layers \
            --stride $stride \
            --batch_size $batch_size \
            --learning_rate $lr \
            --lradj $lradj \
            --enc_in 51 \
            --train_epochs 4 \
            --seq_len 100 \

        done
      done
    done
  done
done
