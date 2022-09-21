# Federated-Handwriting-OCR
This repository provides the code to implement handwriting OCR by federated learning

## Running Environment
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
Parameters are set in lib/config/hw_512_config.yaml  
For federated learning:
* FED.NUM_USERS      the total number of users
* FED.FRAC           the fraction of active users in each communication round
* FED.LOCAL_EPOCH    the number of local training epochs in each communication round
* FED.SHARE_RATE     the fraction of shared data in the whole training data
