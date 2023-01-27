import os

import torch
from tqdm import tqdm

import numpy as np

import clip.clip as clip
#from models.zoo_clip import ClassificationHead
import templates

imr_classnames = ['goldfish', 'great white shark', 'hammerhead', 'stingray', 'hen', 'ostrich', 'goldfinch', 'junco', 
'bald eagle', 'vulture', 'newt', 'axolotl', 'tree frog', 'iguana', 'African chameleon', 'cobra', 'scorpion', 'tarantula', 
'centipede', 'peacock', 'lorikeet', 'hummingbird', 'toucan', 'duck', 'goose', 'black swan', 'koala', 'jellyfish', 'snail', 
'lobster', 'hermit crab', 'flamingo', 'american egret', 'pelican', 'king penguin', 'grey whale', 'killer whale', 'sea lion', 
'chihuahua', 'shih tzu', 'afghan hound', 'basset hound', 'beagle', 'bloodhound', 'italian greyhound', 'whippet', 'weimaraner', 
'yorkshire terrier', 'boston terrier', 'scottish terrier', 'west highland white terrier', 'golden retriever', 'labrador retriever',
'cocker spaniels', 'collie', 'border collie', 'rottweiler', 'german shepherd dog', 'boxer', 'french bulldog', 'saint bernard',
'husky', 'dalmatian', 'pug', 'pomeranian', 'chow chow', 'pembroke welsh corgi', 'toy poodle', 'standard poodle', 'timber wolf',
'hyena', 'red fox', 'tabby cat', 'leopard', 'snow leopard', 'lion', 'tiger', 'cheetah', 'polar bear', 'meerkat', 'ladybug',
'fly', 'bee', 'ant', 'grasshopper', 'cockroach', 'mantis', 'dragonfly', 'monarch butterfly', 'starfish', 'wood rabbit',
'porcupine', 'fox squirrel', 'beaver', 'guinea pig', 'zebra', 'pig', 'hippopotamus', 'bison', 'gazelle', 'llama', 'skunk',
'badger', 'orangutan', 'gorilla', 'chimpanzee', 'gibbon', 'baboon', 'panda', 'eel', 'clown fish', 'puffer fish', 'accordion',
'ambulance', 'assault rifle', 'backpack', 'barn', 'wheelbarrow', 'basketball', 'bathtub', 'lighthouse', 'beer glass',
'binoculars', 'birdhouse', 'bow tie', 'broom', 'bucket', 'cauldron', 'candle', 'cannon', 'canoe', 'carousel', 'castle',
'mobile phone', 'cowboy hat', 'electric guitar', 'fire engine', 'flute', 'gasmask', 'grand piano', 'guillotine', 'hammer',
'harmonica', 'harp', 'hatchet', 'jeep', 'joystick', 'lab coat', 'lawn mower', 'lipstick', 'mailbox', 'missile', 'mitten',
'parachute', 'pickup truck', 'pirate ship', 'revolver', 'rugby ball', 'sandal', 'saxophone', 'school bus', 'schooner',
'shield', 'soccer ball', 'space shuttle', 'spider web', 'steam locomotive', 'scarf', 'submarine', 'tank', 'tennis ball',
'tractor', 'trombone', 'vase', 'violin', 'military aircraft', 'wine bottle', 'ice cream', 'bagel', 'pretzel',
'cheeseburger', 'hotdog', 'cabbage', 'broccoli', 'cucumber', 'bell pepper', 'mushroom', 'Granny Smith', 'strawberry',
'lemon', 'pineapple', 'banana', 'pomegranate', 'pizza', 'burrito', 'espresso', 'volcano', 'baseball player',
'scuba diver', 'acorn']

def get_dnet_classnames():
    cwd = os.getcwd()
    path = os.path.join(cwd,'data/domainnet/anns/clipart_train.txt')
    f  = open(path)
    lines = f.readlines()
    seq = [i.split()[0].split('/')[1] for i in lines]  
    seen = set()
    seen_add = seen.add
    class_list = [x.replace('_',' ').replace('-',' ').lower() for x in seq if not (x in seen or seen_add(x))]
    return class_list

def get_zeroshot_classifier(clip_model,template_style='openai_imagenet_template'):

    template = getattr(templates, template_style)
    logit_scale = clip_model.logit_scale
    
    device = "cuda"
    clip_model.eval()
    clip_model.to(device)
    if 'domainnet' not in template_style:
        classnames = imr_classnames
    else:
        classnames = get_dnet_classnames()
    print('Getting zeroshot weights.')
    with torch.no_grad():
        zeroshot_weights = []
        for classname in tqdm(classnames):
            texts = []
            for t in template:
                texts.append(t(classname))
            texts = clip.tokenize(texts).to(device) # tokenize
            embeddings = clip_model.encode_text(texts) # embed with text encoder
            embeddings /= embeddings.norm(dim=-1, keepdim=True)

            embeddings = embeddings.mean(dim=0, keepdim=True)
            embeddings /= embeddings.norm()

            zeroshot_weights.append(embeddings)

        zeroshot_weights = torch.stack(zeroshot_weights, dim=0).to(device)
        zeroshot_weights = torch.transpose(zeroshot_weights, 0, 2)

        zeroshot_weights *= logit_scale.exp()
        
        zeroshot_weights = zeroshot_weights.squeeze().float()
        zeroshot_weights = torch.transpose(zeroshot_weights, 0, 1)

    #last = ClassificationHead(normalize=True, weights=zeroshot_weights)

    return zeroshot_weights #last