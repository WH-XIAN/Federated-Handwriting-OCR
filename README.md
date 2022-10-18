# Federated-Handwriting-OCR
This repository provides the code to implement handwriting OCR by federated learning.

## Environment
```
    pip3 install -r requirement.txt
```

## Dataset
CASIA (Chinese), Handwriting (English), IAM (English), SCUT-EPT (Chinese), SCUT-HCCDoc (Chinese)  
training data: 662696, testing data: 85303, dictionary: 6636

## Training
```
    CUDA_VISIBLE_DEVICES=0 python3 train_8langs_fed.py 
```

## Parameters
Parameters are defined in *lib/config/hw_512_config.yaml*  
<pre>
* DATASET.LANGUAGE      string, to train on 'Chinese' or 'English' dataset
* MODEL.TYPE            string, to use 'Attention' or 'RNN' as the learning model
* MODEL.NORM            string, to use 'BatchNorm' or 'LayerNorm' as normalization
* FED.NUM_USERS         int, the total number of users
* FED.FRAC              float, the fraction of active users in each communication round
* FED.LOCAL_EPOCH       int, the number of local training epochs in each communication round
* FED.SHARE_RATE        float, the fraction of shared data in the whole training dataset
* FED.DC                bool, whether to use drift control
* FED.DC_ALPHA          float, the coefficient of drift control term
</pre>
* To train RNN on Chinese + English dataset, set  
*DATASET.LANGUAGE='Chinese'*  
*MODEL.TYPE='RNN'*    
*DATASET.JSON_FILE['train']=* the path of *hw_train.json*  
*DATASET.JSON_FILE['val']=* the path of *hw_test.json*  
* To train Self Attention on English dataset, set  
*DATASET.LANGUAGE='English'*   
*MODEL.TYPE='Attention'*   
*DATASET.JSON_FILE['train']=* the path of *hw_train_en.json*   
*DATASET.JSON_FILE['val']=* the path of *hw_test_en.json*    

**Remember to change** the path of dictionary in *lib/config/alphabets_new.py*  
The number of communication rounds is *TRAIN.END_EPOCH*. Learning rate can be tuned by changing *TRAIN.LR*.
