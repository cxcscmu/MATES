python src/pretrain/pretrain.py \
    --model_name $model_name \
    --method $method \
    --ckpt $ckpt \
    --data_path data/c4/$model_name/$method/$ckpt \
    --out_path out/c4/$model_name/$method \
    --decay $decay