# ROSE: Toward Reality-Oriented Safety Evaluation of Large Lanuage Models

## Setup

1. Start by installing the packages needed 

```
conda create -n rose python=3.10 
conda activate rose
pip install -r requirements.txt
```

2. config your ```accelerate``` by running 
```
accelerate config
```
, then add the file path to ```Reality-Oriented-Safety-Evaluation/bash_scripts\XXX.sh```. You need at least **3** GPUs with 24 GiB memory each to finish all the experiments.

3. Provide your api-key for ```Aliyun```, ```openai```, ```Gemini```, and ```Deepseek``` in the script ```Reality-Oriented-Safety-Evaluation/supplementary_models.py``` and ```Reality-Oriented-Safety-Evaluation/utils/api_generation.py```, where the places have been marked by ```<YOUR API>```.

## Start your Reality-Oriented Safety Evaluation!
* Run the bash script
```
bash Reality-Oriented-Safety-Evaluation/bash_scripts/baseline_benchmark.sh
```
to evaluate the baseline benchmarks against LLMs.

* Run the bash script
```
bash Reality-Oriented-Safety-Evaluation/bash_scripts/baseline_RFT.sh
```
to evaluate the baseline RFT-based methods against LLMs.




