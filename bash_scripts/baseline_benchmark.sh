cd ~

batch_size=128
col_name='prompt'
iteration_num=80
jailbreak_template="False"

for seed in {1000..3000..1000}; do
    for data_path in "Reality-Oriented-Safety-Evaluation/data/JailBreakV_28K/JailBreakV_28K.xlsx" "Reality-Oriented-Safety-Evaluation/data/latent-jailbreak/letent.csv" "Reality-Oriented-Safety-Evaluation/data/wild-jailbreak/wild_jailbreak.xlsx" "Reality-Oriented-Safety-Evaluation/data/jailbreak_llms/jailbreak_llms.xlsx"; do
        for system_prompt in "False" "True"; do
            for victim_model in "qwen-turbo" "gpt4o" "gemini" "deepseek"; do
                echo '...seed='$seed
                echo '...data_path='$data_path
                echo '...victim_model='$victim_model
                echo '...use system prompt='$system_prompt

                accelerate launch --config_file <ACCELERATE-CONFIG-FILE> \
                Reality-Oriented-Safety-Evaluation/scripts/baseline_Dataset.py \
                --data_path ${data_path} --col_name ${col_name} \
                --batch_size ${batch_size} --iteration_num ${iteration_num} \
                --seed ${seed} --system_prompt ${system_prompt} --victim_model ${victim_model} \
                > Reality-Oriented-Safety-Evaluation/logs/outputs/Dataset_${seed}_${data_path##*/}_${system_prompt}_${victim_model}.txt
            done
        done    
    done
done

for seed in {1000..3000..1000}; do
    for data_path in "Reality-Oriented-Safety-Evaluation/data/JailBreakV_28K/JailBreakV_28K.xlsx" "Reality-Oriented-Safety-Evaluation/data/latent-jailbreak/letent.csv" "Reality-Oriented-Safety-Evaluation/data/wild-jailbreak/wild_jailbreak.xlsx" "Reality-Oriented-Safety-Evaluation/data/jailbreak_llms/jailbreak_llms.xlsx"; do
        for system_prompt in "True"; do
            for victim_model in "open_gemma" "open_llama" "open_deepseek"; do
                echo '...seed='$seed
                echo '...data_path='$data_path
                echo '...victim_model='$victim_model
                echo '...use system prompt='$system_prompt

                accelerate launch --config_file <ACCELERATE-CONFIG-FILE> \
                Reality-Oriented-Safety-Evaluation/scripts/baseline_Dataset.py \
                --data_path ${data_path} --col_name ${col_name} \
                --batch_size ${batch_size} --iteration_num ${iteration_num} \
                --seed ${seed} --system_prompt ${system_prompt} --victim_model ${victim_model} \
                > Reality-Oriented-Safety-Evaluation/logs/outputs/Dataset_${seed}_${data_path##*/}_${system_prompt}_${victim_model}.txt
            done
        done    
    done
done
