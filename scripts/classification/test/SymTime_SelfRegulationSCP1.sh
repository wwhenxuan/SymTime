export CUDA_VISIBLE_DEVICES=4

for forward_layers in 3 1 2
do
  for batch_size in 16 4 8 12
  do
    for lr in 0.0001 0.0002 0.0003 0.0004 0.0005
    do
      for aj in "type1" "type2" "cosine"
      do
        python -u classification.py \
          --task_name "classification" \
          --dataset_name "SelfRegulationSCP1" \
          --data "UEA" \
          --root_path "./datasets/SelfRegulationSCP1" \
          --forward_layers $forward_layers \
          --out_channels 16 \
          --stride 16 \
          --batch_size $batch_size \
          --learning_rate $lr \
          --num_workers 0 \
          --lradj $aj \
          --patience 8 \

      done
    done
  done
done
