# Discovering State Variables Hidden for FHN Equation



## MultiProcessing Script

Start 20 subprocesses for [generate data -- train -- eval -- calculate ID] in different tau of 10 random seeds

```shell
cd FHN/
python pipeline_multiprocessing.py
```



---



## Data Preparation

```shell
cd FHN/Data
python3 0_data_gen.py
python3 2_create_training_data.py
```

## Training and Testing

```shell
cd FHN/
python fhn_main.py
export MPLBACKEND=Agg
python fhn_eval.py
```

## Intrinsic Dimension Estimation

```shell
cd FHN
python fhn_eval_intrinsic_dimension.py
PYTHONIOENCODING=utf-8 python dimension.py
```
