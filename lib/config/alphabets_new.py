import json
import os

with open('/home/wex37/FedOCR/data/hw_dict.json', 'r') as f:
    dic_cn = json.load(f)
alphabet_cn = ''.join(list(dic_cn.keys()))

