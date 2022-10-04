# Federated-Handwriting-OCR
This repository provides the code to implement handwriting OCR by federated learning.

## Environment
```
    pip3 install -r requirement.txt
```

## Training
```
    CUDA_VISIBLE_DEVICES=0 python3 train_8langs_fed.py 
```

## Dataset
CASIA (Chinese), Handwriting (English), IAM (English), SCUT-EPT (Chinese), SCUT-HCCDoc (Chinese)  
training data: 662696, testing data: 85303, dictionary: 6636

## Parameters
Parameters are defined in *lib/config/hw_512_config.yaml*  
* MODEL.NORM &emsp;&emsp;&emsp;&emsp;&emsp; &ensp;       *string*, to use 'BatchNorm' or 'LayerNorm'
* FED.NUM_USERS &emsp;&emsp;&emsp; &emsp;      *int*, the total number of users
* FED.FRAC   &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;         *float*, the fraction of active users in each communication round
* FED.LOCAL_EPOCH &emsp;&emsp;&ensp; &ensp;    *int*, the number of local training epochs in each communication round
* FED.SHARE_RATE  &emsp;&emsp;&emsp;&emsp;    *float*, the fraction of shared data in the whole training data
* FED.STARTUP_EPOCH &emsp;&ensp;  *int*, the number of epochs for pretraining on shared data
* FED.DC     &emsp;&emsp;&emsp;&emsp; &emsp; &emsp;&emsp; &emsp;     *bool*, whether to use drift correction
* FED.DC_ALPHA  &emsp; &emsp; &emsp; &emsp;      *float*, the coefficient of drift correction term
