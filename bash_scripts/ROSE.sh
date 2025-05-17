cd ~

lr_rate=0.000005
batch_size=64
mini_batch_size=8
ppo_epochs=4
col_name='question'
iteration_num=160
jailbreak_template="False"
div_threshold=0.4

for seed in {1000..3000..1000}; do
    for data_path in "ROSE/data/reddit_tifu/tifu.csv"; do
        for system_prompt in "False" "True"; do
            for victim_model in "qwen-turbo" "gpt4o" "gemini" "deepseek"; do
                echo '...seed='$seed
                echo '...data_path='$data_path
                echo '...victim_model='$victim_model
                echo '...use system prompt='$system_prompt

                accelerate launch --config_file <ACCELERATE-CONFIG-FILE> \
                Reality-Oriented-Safety-Evaluation/scripts/ROSE.py \
                --data_path ${data_path} --col_name ${col_name} \
                --lr_rate ${lr_rate} --batch_size ${batch_size} --mini_batch_size ${mini_batch_size} --ppo_epochs ${ppo_epochs} --iteration_num ${iteration_num} \
                --seed ${seed} --system_prompt ${system_prompt} --victim_model ${victim_model} --jailbreak_template ${jailbreak_template} \
                --div_threshold ${div_threshold} \
                > Reality-Oriented-Safety-Evaluation/logs/outputs/ROSE_${seed}_${data_path##*/}_${system_prompt}_${victim_model}.txt
            done
        done    
    done
done

for seed in {1000..3000..1000}; do
    for data_path in "ROSE/data/reddit_tifu/tifu.csv"; do
        for system_prompt in "False"; do
            for victim_model in "open_gemma" "open_llama" "open_deepseek"; do
                echo '...seed='$seed
                echo '...data_path='$data_path
                echo '...victim_model='$victim_model
                echo '...use system prompt='$system_prompt

                accelerate launch --config_file <ACCELERATE-CONFIG-FILE> \
                Reality-Oriented-Safety-Evaluation/scripts/ROSE.py \
                --data_path ${data_path} --col_name ${col_name} \
                --lr_rate ${lr_rate} --batch_size ${batch_size} --mini_batch_size ${mini_batch_size} --ppo_epochs ${ppo_epochs} --iteration_num ${iteration_num} \
                --seed ${seed} --system_prompt ${system_prompt} --victim_model ${victim_model} --jailbreak_template ${jailbreak_template} \
                --div_threshold ${div_threshold} \
                > Reality-Oriented-Safety-Evaluation/logs/outputs/ROSE_${seed}_${data_path##*/}_${system_prompt}_${victim_model}.txt
            done
        done    
    done
done
