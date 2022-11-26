# myLED

> Reference：[cselab/LED (github.com)](https://github.com/cselab/LED)


The scripts to generate the training, validation, and test data for each application can be found in the ./LED/Data folder.
Run these scripts in the respective order, i.e. for the FHN equation:



```shell
# generator train and val dataset
python3 0_data_gen.py
python3 2_create_training_data.py

# plot train dataset
# python3 1_creating_figures.py

# generator test dataset
python3 3_data_gen_test.py
python3 4_create_test_data.py
```
Note: 
* sample time length == subsampling * dt (dt=5ms) ------ [ in 0_data_gen.py and 3_data_gen_test.py ]
* time series length == N_TRAIN --------------------------- [ in 2_create_training_data.py ]