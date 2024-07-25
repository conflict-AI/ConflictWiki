from wb0configs import configs
from wb0configs.helpers import store_file, load_file

import re


def change_entity_name2id(text, entity_dict):
    entity_dict_r = {v: k for k,v in entity_dict.items()}
    text = re.sub(r"(<)(.+?)(/>)", lambda m: '<{}/>'.format(entity_dict_r.get(m.group(2))), text)  ## consider entities in [[ ]]
    return text


