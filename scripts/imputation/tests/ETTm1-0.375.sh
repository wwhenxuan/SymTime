export CUDA_VISIBLE_DEVICES=6

for batch_size in 4 8 12 16 20 24
do
  for forward_layers in 2 3 4
  do
    for stride in 1 4 8
    do
      for lradj in "type1" "type2" "cosine"
      do
        for lr in 0.0001 0.0002 0.0003 0.0004 0.0005
        do
          python -u imputation.py \
            --task_name "imputation" \
            --dataset_name "ETTm1" \
            --data "ETTm1" \
            --root_path "./datasets/ETT/" \
            --data_path "ETTm1.csv" \
            --mask_rate 0.375 \
            --forward_layers $forward_layers \
            --stride $stride \
            --batch_size $batch_size \
            --learning_rate $lr \
            --lradj $lradj \

        done
      done
    done
  done
done
