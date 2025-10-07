export CUDA_VISIBLE_DEVICES=4


python -u classification.py \ 
  --task_name "classification" \ 
  --dataset_name "SelfRegulationSCP1" \ 
  --data "UEA" \ 
  --root_path "./datasets/SelfRegulationSCP1" \ 
  --forward_layers 3 \ 
  --out_channels 16 \ 
  --stride 16 \ 
  --batch_size 12 \ 
  --learning_rate 0.0005 \ 
  --num_workers 0 \ 
  --lradj "cosine" \ 
  --patience 8 \ 
  