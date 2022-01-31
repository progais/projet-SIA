# SIA project 
## SOUKOUNA Aisata

# English to French translation with a streamlit tutorial

## Quick Start Streamlit tutorial
Run : streamlit run streamlit_NLP.py

## Quick Start NLP translation
```
Configuration : `./experiments/attention/f2e/config.json'`
```
```
Run : python main.py --gpu 0 --model_dir experiment/attention/f2e/
```
```
Model : `./experiment/attention/f2e/ckpts/best.pth.tar`. 
```
```
Test : python main.py --gpu 0 --model_dir experiment/attention/f2e/ --mode test --heatmap
```

## How to use

### Training 
python main.py --gpu [gpu_id] --model_dir [model_dir]
### Test
python main.py --gpu [gpu_id] --model_dir [model_dir] --mode test
Optional arguments:
- `--heatmap` or `--hm` whether to generate and store the attention weight heat map.

