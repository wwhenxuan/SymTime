export CUDA_VISIBLE_DEVICES=0

for batch_size in 24 20 16 12 8 4
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
            --dataset_name "ETTh1" \
            --data "ETTh1" \
            --root_path "./datasets/ETT/" \
            --data_path "ETTh1.csv" \
            --mask_rate 0.25 \
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
