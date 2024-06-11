gpu_index=0
for s in {0..7}; do
    echo $s
    CUDA_VISIBLE_DEVICES=$gpu_index nohup python src/select_data/predict_data_influence.py --model_name $model_name --ckpt $ckpt --shard $s 8 > log_job_s${s}_gpu${gpu_index}.out 2>&1 &
    ((gpu_index=(gpu_index+1)%8))
done