export CUDA_VISIBLE_DEVICES=0

forward_layers=3
stride=1
batch_size=4
lr=0.0005
lradj="type1"


python -u imputation.py \
  --task_name "imputation" \
  --dataset_name "ECL" \
  --data "custom" \
  --root_path "./datasets/electricity/" \
  --data_path "electricity.csv" \
  --mask_rate 0.125 \
  --forward_layers 2 \
  --stride $stride \
  --batch_size $batch_size \
  --learning_rate 0.00025 \
  --lradj "cosine" \


python -u imputation.py \
  --task_name "imputation" \
  --dataset_name "ECL" \
  --data "custom" \
  --root_path "./datasets/electricity/" \
  --data_path "electricity.csv" \
  --mask_rate 0.25 \
  --forward_layers $forward_layers \
  --stride $stride \
  --batch_size $batch_size \
  --learning_rate $lr \
  --lradj $lradj \


python -u imputation.py \
  --task_name "imputation" \
  --dataset_name "ECL" \
  --data "custom" \
  --root_path "./datasets/electricity/" \
  --data_path "electricity.csv" \
  --mask_rate 0.375 \
  --forward_layers $forward_layers \
  --stride $stride \
  --batch_size $batch_size \
  --learning_rate $lr \
  --lradj $lradj \


python -u imputation.py \
  --task_name "imputation" \
  --dataset_name "ECL" \
  --data "custom" \
  --root_path "./datasets/electricity/" \
  --data_path "electricity.csv" \
  --mask_rate 0.50 \
  --forward_layers $forward_layers \
  --stride $stride \
  --batch_size $batch_size \
  --learning_rate $lr \
  --lradj "type2" \
