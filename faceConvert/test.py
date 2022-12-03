import json
from collections import OrderedDict

with open('./mywebsite/static/object/data/info.json', 'r') as f:
    json_data = json.load(f)


print(json_data)
model_list = json_data['list']
new_json = {}
new_json['list'] = []



for model in model_list:

    model_data = {}

    model_data['name'] = model['name']
    model_data['front-R'] = model['front-R']
    if model['front-L'] == 'same':
        model_data['front-L'] = model['front-R']
    else:
        model_data['front-L'] = model['front-L']
    model_data['base-set'] = model['base-set']

    new_json['list'].append(model_data)

    if model['reverse'] == True:
        model_data = {}
        model_data['name'] = model['name'] +'r'
        model_data['front-L'] = model['front-R']
        if model['front-L'] == 'same':
            model_data['front-R'] = model['front-R']
        else:
            model_data['front-R'] = model['front-L']
        model_data['base-set'] = model['base-set']

        new_json['list'].append(model_data)

print(new_json)