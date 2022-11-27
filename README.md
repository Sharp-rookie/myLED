# Discovering State Variables Hidden for FHN Equation

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
python fhn_eval.py
```

## Intrinsic Dimension Estimation

```shell
cd FHN
python fhn_eval_intrinsic_dimension.py
python dimension.py
```
