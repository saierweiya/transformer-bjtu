
export PYTHONHASHSEED=42

mkdir -p results

echo "Running standard (Large) Transformer..."
python src/main.py \
    --data_path "./data" \
    --batch_size 16 \
    --d_model 512 \
    --nhead 8 \
    --num_encoder_layers 6 \
    --num_decoder_layers 6 \
    --dim_feedforward 2048 \
    --dropout 0.1 \
    --epochs 30 \
    --lr 0.0003 \
    --max_samples 30000 \
    --max_length 64 \
    --save_path "./results/standard_Transformer" \
    --seed 42 \
    --ablation_type "full"

# max_samples 开发和调试阶段、消融研究 (Ablation Study)、计算资源限制、超参数调优

if [ $? -eq 0 ]; then
    echo "Standard (Large) Transformer training completed successfully."

    echo "Running Small Transformer (ablation study)..."
    python src/main.py \
        --data_path "./data" \
        --batch_size 16 \
        --d_model 256 \
        --nhead 4 \
        --num_encoder_layers 3 \
        --num_decoder_layers 3 \
        --dim_feedforward 1024 \
        --dropout 0.1 \
        --epochs 30 \
        --lr 0.0003 \
        --max_samples 30000 \
        --max_length 64 \
        --save_path "./results/Small_Transformer" \
        --seed 42 \
        --use_small_model \
        --ablation_type "full"

    if [ $? -eq 0 ]; then
        echo "Ablation study (Small model) completed successfully.  消融实验成功结束"

        echo "Plotting size ablation comparison results..."
        python src/plot_results.py

        echo "Size ablation study and comparison plotting completed! Compare results in ./results/"
    else
        echo "Ablation study (Small model) failed."
    fi
else
    echo "Standard (Large) Transformer training failed."
fi


echo "开始对四个模块进行消融实验：层归一化、无位置编码、无残差、单多头注意力......."

echo "无残差连接消融实验。。"
python src/main.py \
    --data_path "./data" \
    --batch_size 16 \
    --d_model 256 \
    --nhead 4 \
    --num_encoder_layers 3 \
    --num_decoder_layers 3 \
    --dim_feedforward 1024 \
    --dropout 0.1 \
    --epochs 30 \
    --lr 0.0003 \
    --max_samples 30000 \
    --max_length 64 \
    --save_path "./results/no_residual_Small_Transformer" \
    --seed 42 \
    --use_small_model \
    --ablation_type "no_residual"

echo "无位置编码消融实验。。"
python src/main.py \
    --data_path "./data" \
    --batch_size 16 \
    --d_model 256 \
    --nhead 4 \
    --num_encoder_layers 3 \
    --num_decoder_layers 3 \
    --dim_feedforward 1024 \
    --dropout 0.1 \
    --epochs 30 \
    --lr 0.0003 \
    --max_samples 30000 \
    --max_length 64 \
    --save_path "./results/no_positional_encoding_Small_Transformer" \
    --seed 42 \
    --use_small_model \
    --ablation_type "no_positional_encoding"

echo "无层归一化消融实验。。"
python src/main.py \
    --data_path "./data" \
    --batch_size 16 \
    --d_model 256 \
    --nhead 4 \
    --num_encoder_layers 3 \
    --num_decoder_layers 3 \
    --dim_feedforward 1024 \
    --dropout 0.1 \
    --epochs 30 \
    --lr 0.0003 \
    --max_samples 30000 \
    --max_length 64 \
    --save_path "./results/no_layer_norm_Small_Transformer" \
    --seed 42 \
    --use_small_model \
    --ablation_type "no_layer_norm"

echo "单头注意力消融实验。。"
python src/main.py \
    --data_path "./data" \
    --batch_size 16 \
    --d_model 256 \
    --nhead 1 \
    --num_encoder_layers 3 \
    --num_decoder_layers 3 \
    --dim_feedforward 1024 \
    --dropout 0.1 \
    --epochs 30 \
    --lr 0.0003 \
    --max_samples 30000 \
    --max_length 64 \
    --save_path "./results/single_head_Small_Transformer" \
    --seed 42 \
    --use_small_model
