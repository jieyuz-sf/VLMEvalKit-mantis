from PIL import Image
import json, os
from tqdm import tqdm
p = '/export/share/ayan/data/json_jieyu/attr_v3_train.json'
img_folder = '/export/share/ayan/data'
a = json.load(open(p, 'r'))
newa = []
for c in tqdm(a):
    try:
        Image.open(os.path.join(img_folder, c['image'])).convert('RGB')
    except:
        print(c)
        continue
    newa.append(c)
print(len(newa))
json.dump(newa, open(p, 'w'), indent=2)
