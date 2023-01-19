import json

data = json.load(open('./imr_class_map.json'))
inverse_map = {}

for key,val in data.items():
    inverse_map[val] = key

json.dump(inverse_map,open('./imr_class_reverse_map.json','w'))