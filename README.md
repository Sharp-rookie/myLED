# myLED

> Reference：[cselab/LED (github.com)](https://github.com/cselab/LED)


The scripts to generate the training, validation, and test data for each application can be found in the ./LED/Data folder.
Run these scripts in the respective order, i.e. for the FHN equation:

```shell
python3 0_data_gen.py
# python3 1_creating_figures.py
python3 2_create_training_data.py
python3 3_data_gen_test.py
python3 4_create_test_data.py
```


Train and test AE-LSTM model:

```shell
./AE-LSTM.sh
```

Train and test AE-LSTM model with only inhibitor input:

```shell
./AE-LSTM-inhibitor.sh
```

Train and test CNN-LSTM model:

```shell
./CNN-LSTM.sh
```

Train and test CNN-LSTM model with only inhibitor input:

```shell
./CNN_LSTM-inhibitor.sh
```