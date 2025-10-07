export CUDA_VISIBLE_DEVICES=2

for forward_layers in 3 1 2
do
  for batch_size in 16 32 48 64
  do
    for lr in 0.0001 0.0002 0.0003 0.0004 0.0005
    do
      for aj in "type1" "type2" "cosine"
      do
        python -u classification.py \
          --task_name "classification" \
          --dataset_name "SpokenArabicDigits" \
          --data "UEA" \
          --root_path "./datasets/SpokenArabicDigits" \
          --forward_layers $forward_layers \
          --out_channels 13 \
          --stride 4 \
          --batch_size $batch_size \
          --learning_rate $lr \
          --num_workers 0 \
          --lradj $aj \
          --patience 16 \

      done
    done
  done
done
