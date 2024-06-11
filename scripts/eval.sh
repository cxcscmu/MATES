mkdir -p results/c4/$model_name/$method/iter-$ckpt-ckpt

python src/pretrain/lm_eval_harness.py \
        --checkpoint_dir "out/c4/$model_name/$method/iter-$ckpt-ckpt.pth" \
        --tokenizer_dir "tokenizer/togethercomputer/RedPajama-INCITE-Base-7B-v0.1" \
        --model_name "$model_name-1024" \
        --num_fewshot 0 \
        --eval_tasks "[sciq,arc_easy,arc_challenge,logiqa,openbookqa,boolq,hellaswag,piqa,winogrande]" \
        --precision "bf16-true" \
        --batch_size 4 \
        --save_filepath "results/c4/$model_name/$method/iter-$ckpt-ckpt/results.json"