# Discovering Slow Variables Hidden for FHN Equation



## MultiProcessing Script

Start multi-subprocesses for different tau of 5 random seeds

1. Generate data in 30 traces with the initial conditions added Gaussian noise
2. Train time-lagged on 29 trace and test on 1 trace
3. calculate the IDs with the embedding of test data

```shell
python pipeline_multiprocessing.py
```
