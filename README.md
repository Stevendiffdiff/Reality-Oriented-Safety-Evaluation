# <img src='sourced-materials/logo.png' width="15%"> <font color=deeppink>ROSE<font color=lightgray>: Toward Reality-Oriented Safety Evaluation of Large Lanuage Models

## What is reality-oriented safety evaluation for LLMs?

### [1] Adaptive generation
The evaluation method is supposed to generate effective adversarial prompts that keep abreast with the SOTA LLMs.
### [2] Topically diverse
The prompts contains diverse harmful goals:
* make a bomb
* assassinate a person
* generate sexual contents
* ...

**Are existing methods topically diverse?**
<img src='sourced-materials/tab-diversity.png'>

### [3] Contextualized
The prompts are grounded in real-world scenarios:
* sent my girlfriend to work an hour early
* drinking a whole bottle of black cherry juice
* ...

<img src='sourced-materials/tab-consistency.png'>

## What is <font color=deeppink>ROSE<font color=lightgray>?
Our method includes three key components:
1. <font color="004C99"> **contextualized adversarial prompt generation** <font color=lightgray> pipeline;
2. <font color="#1F6D42"> **reality-oriented reward system** <font color=lightgray>;
3. <font color="#FF914E"> **multi-objective RFT-based optimization** <font color=lightgray>. 

<img src="sourced-materials/ROSE-framework.png">

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

* Run the bash script
```
bash Reality-Oriented-Safety-Evaluation/bash_scripts/ROSE.sh
```
to evaluate the ROSE against LLMs.

## ROSEset
You can get our out-of-the-box dataset ```ROSEset``` in ```Reality-Oriented-Safety-Evaluation/data/ROSEset```, where we provide three versions of the benchmark. You are recommended to use ```Reality-Oriented-Safety-Evaluation/data/ROSEset/benchmark-v2.xlsx```, which yields the best performance in our experiments.

<img src="sourced-materials/ROSEset-radar.png">




