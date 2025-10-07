export CUDA_VISIBLE_DEVICES=3

for forward_layers in 1 2
do
  for batch_size in 32 16 48 64
  do
    for lr in 0.0001 0.0002 0.0003 0.0004 0.0005
    do
      for aj in "type1" "type2" "cosine"
      do
        python -u classification.py \
          --task_name "classification" \
          --dataset_name "FaceDetection" \
          --data "UEA" \
          --root_path "./datasets/FaceDetection" \
          --forward_layers $forward_layers \
          --out_channels 32 \
          --stride 1 \
          --batch_size $batch_size \
          --learning_rate $lr \
          --num_workers 0 \
          --lradj $aj \

      done
    done
  done
done
