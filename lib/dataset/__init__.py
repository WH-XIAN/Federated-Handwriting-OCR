from ._360cc import _360CC
from ._own import _OWN
from ._cn import _Cn
# from ._icdar import _icdar
from ._latin import _Latin
# from ._xiaotiancai import _Xiaotiancai
def get_dataset(config):

    if config.DATASET.DATASET == "360CC":
        return _360CC
    elif config.DATASET.DATASET == "OWN":
        return _OWN
    elif config.DATASET.DATASET == "JpKr_new":
        return _Latin
    elif config.DATASET.DATASET == "icdar":
        return _icdar
    elif config.DATASET.DATASET == "Latin":
        return _Latin
    elif config.DATASET.DATASET == "8Langs":
        return _Latin
    elif config.DATASET.DATASET == "Hindi":
        return _Latin
    elif config.DATASET.DATASET == "10langs":
        return _Latin
    # elif config.DATASET.DATASET.startswith("Xiaotiancai"):
    #     return _Xiaotiancai
    elif config.DATASET.DATASET == "Ch&En": 
        return _Cn # 原来是使用  _Latin
    elif config.DATASET.DATASET == "Hw_Ch&En": 
        return _Latin
    else:
        raise NotImplemented()
    
    
# def get_dataset(config):
    
#     if config.DATASET.DATASET == "360CC":
#         return _360CC
#     elif config.DATASET.DATASET == "OWN":
#         return _OWN
#     elif config.DATASET.DATASET == "JpKr_new":
#         return _Latin
#     elif config.DATASET.DATASET == "icdar":
#         return _icdar
#     elif config.DATASET.DATASET == "Latin":
#         return _Latin
#     elif config.DATASET.DATASET == "8Langs":
#         return _Latin
#     elif config.DATASET.DATASET == "Hindi":
#         return _Latin
#     elif config.DATASET.DATASET == "10langs":
#         return _Latin
#     # elif config.DATASET.DATASET.startswith("Xiaotiancai"):
#     #     return _Xiaotiancai
#     elif config.DATASET.DATASET == "Ch&En": 
#         return _Latin
#     else:
#         raise NotImplemented()