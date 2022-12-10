# myLED

> Reference：[cselab/LED (github.com)](https://github.com/cselab/LED)


The scripts to generate the training, validation, and test data for each application can be found in the ./LED/Data folder.
Run these scripts in the respective order, i.e. for the FHN equation:



```shell
# generator train, val and test dataset
python3 0_data_gen.py
python3 2_create_training_data.py

# plot train dataset
# python3 1_creating_figures.py
```
Note: 
* sample time length == subsampling * dt (dt=5ms) ------ [ in 0_data_gen.py ]
* total time series length == total_length --------------------------- [ in 2_create_training_data.py ]
* each sample length == sequence_length --------------------------- [ in 2_create_training_data.py ]