export CUDA_VISIBLE_DEVICES=3


python -u classification.py \
  --task_name "classification" \
  --dataset_name "PEMS-SF" \
  --data "UEA" \
  --root_path "./datasets/PEMS-SF" \
  --forward_layers 2 \
  --conv1d True \
  --out_channels 32 \
  --stride 4 \
  --batch_size 32 \
  --learning_rate 0.0001 \
  --num_workers 0 \
  --lradj "type1" \
  --patience 10 \