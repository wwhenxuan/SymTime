export CUDA_VISIBLE_DEVICES=5


python -u anomaly_detection.py \
            --task_name "anomaly_detection" \
            --dataset_name "SMD" \
            --data "SMD" \
            --root_path "./datasets/SMD" \
            --anomaly_ratio 0.5 \
            --forward_layers 3 \
            --stride 8 \
            --batch_size 32 \
            --learning_rate 0.0005 \
            --lradj "type1" \
            --enc_in 38 \
            --train_epochs 12 \
            --seq_len 100 \
