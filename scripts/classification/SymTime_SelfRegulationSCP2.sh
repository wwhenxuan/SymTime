export CUDA_VISIBLE_DEVICES=2


python -u classification.py \ 
  --task_name "classification" \ 
  --dataset_name "SelfRegulationSCP2" \ 
  --data "UEA" \ 
  --root_path "./datasets/SelfRegulationSCP2" \ 
  --forward_layers 1 \ 
  --out_channels 7 \ 
  --stride 16 \ 
  --batch_size 8 \ 
  --learning_rate 0.0001 \ 
  --num_workers 0 \ 
  --lradj "type1" \ 
  --patience 8 \ 