export CUDA_VISIBLE_DEVICES=5

for forward_layers in 3 1 2
do
  for batch_size in 32 4 8 12 16
  do
    for lr in 0.0001 0.0002 0.0003 0.0004 0.0005
    do
      for aj in "type1" "type2" "cosine"
      do
        for out_channels in 16 32 64
        do
          for stride in 4 8
          do
            python -u classification.py \
              --task_name "classification" \
              --dataset_name "UWaveGestureLibrary" \
              --data "UEA" \
              --root_path "./datasets/UWaveGestureLibrary" \
              --forward_layers $forward_layers \
              --out_channels $out_channels \
              --stride $stride \
              --batch_size $batch_size \
              --learning_rate $lr \
              --num_workers 0 \
              --lradj $aj \

          done
        done
      done
    done
  done
done
