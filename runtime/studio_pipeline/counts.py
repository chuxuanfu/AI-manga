"""Read explicit quantities without depending on the reviewer's choice of category."""
import re
from collections import defaultdict

# Canonical nouns match the detector and the independent image inventory.
ALIASES={
 'person':'person people human humans girl girls boy boys man men woman women child children adult adults 女孩 少女 女孩儿 男孩 男生 女生 小朋友 儿童 男子 女子 男人 女人 人物',
 'cat':'cat cats kitten kittens kitty 猫 小猫 花猫 橘猫', 'dog':'dog dogs puppy puppies shiba 犬 狗 柴犬',
 'rabbit':'rabbit rabbits bunny bunnies 兔 白兔 兔子', 'fox':'fox foxes 狐狸 狐', 'bird':'bird birds 鸟',
 'horse':'horse horses 马', 'bear':'bear bears 熊', 'turtle':'turtle turtles tortoise tortoises 龟 海龟 乌龟 烏龜 龜','bicycle':'bicycle bicycles bike bikes 自行车 单车',
 'umbrella':'umbrella umbrellas parasol parasols 伞 阳伞', 'teapot':'teapot teapots 茶壶', 'car':'car cars 汽车 轿车',
 'train':'train trains 列车 火车', 'boat':'boat boats 船', 'cup':'cup cups mug mugs 杯子 茶杯',
 'book':'book books 书 书本', 'animal':'animal animals 动物'}
NUMBERS={'zero':0,'no':0,'one':1,'a':1,'an':1,'two':2,'three':3,'four':4,'five':5,'six':6,'seven':7,'eight':8,'nine':9,
         '十':10,'一':1,'二':2,'两':2,'三':3,'四':4,'五':5,'六':6,'七':7,'八':8,'九':9}
english={a:key for key,words in ALIASES.items() for a in words.split() if a.isascii()}
chinese={a:key for key,words in ALIASES.items() for a in words.split() if not a.isascii()}
boundaries=set('at in beside across from on of and or behind holding with under sits is shows by next near to sleeping asleep inside outside'.split())
animals=set('cat dog rabbit fox bird horse bear sheep cow elephant zebra giraffe deer squirrel otter panda wolf tiger lion turtle'.split())


def inventory_counts(items):
    """Compare counts at the same taxonomic level while retaining breed in the visual evidence."""
    counts={}
    species=set()
    varieties=defaultdict(dict)
    for item in items:
        label=item['category'].strip().lower()
        n=item['count']
        if not label or type(n) is not int or not 0<=n<=100:
            continue
        if item['kind']=='person':
            counts['person']=max(n,counts.get('person',0))
        elif item['kind']=='animal':
            reported=item.get('counting_species','').strip().lower()
            name=reported if reported not in ('','unknown','animal','animals') else label
            from animal_identity import domestic_species, species_noun
            label=species_noun(label)
            name=species_noun(name)
            name=domestic_species(label) or domestic_species(name) or english.get(name,name)
            if name not in ('animal','animals','unknown'):
                species.add(name)
            varieties[name][label]=max(n,varieties[name].get(label,0))
        else:
            counts[english.get(label,label)]=n
    for name,labels in varieties.items():
        # A total such as "dog: 2" can accompany independently named breeds.
        counts[name]=max(labels.get(name,0),sum(n for label,n in labels.items() if label!=name))
    return counts,species


def explicit_counts(text):
    values=defaultdict(list)
    words=re.findall(r"[a-z]+|[0-9]+|[^\w\s]",text.lower())
    for i,word in enumerate(words):
        if word not in NUMBERS and not word.isdigit():continue
        for noun in words[i+1:i+6]:
            if noun in english:
                values[english[noun]].append(int(word) if word.isdigit() else NUMBERS[word]);break
            if not noun.isalpha() or noun in boundaries or noun in NUMBERS:break
    names='|'.join(map(re.escape,sorted(chinese,key=len,reverse=True)))
    for match in re.finditer(r'([一二两三四五六七八九十]|[0-9]+)\s*[位只把匹头名辆艘本个]\s*([^，。；、！？,.]{0,10}?)('+names+')',text):
        n,_,noun=match.groups()
        values[chinese[noun]].append(int(n) if n.isdigit() else NUMBERS[n])
    # Repeated noun mentions can refer to the same individual; leave those to the semantic extraction.
    return {name:ns[0] for name,ns in values.items() if len(ns)==1}


def reconcile(generated,source,observed=False):
    negative=bool(re.search(r'\b(no|none|zero|without|absent)\b|没有|无人|无人物|不含|不要|未出现',source,re.I))
    counts={name:n for name,n in generated.items() if n or negative}
    # A scene-level figurine still represents its subject. Conversely a portrait on
    # the wall does not add a second person. Preserve the semantic reading of these
    # clauses rather than overriding it with a regex over "no real people".
    layered=observed and re.search(r"\b(dolls?|figurines?|portraits?|reflection[s]?|miniatures?|dioramas?|illustrations?|puppets?|clay|felt|printed|painted)\b",source,re.I)
    if not layered:counts.update(explicit_counts(source))
    if 'animal' in counts and any(s in counts for s in animals):counts.pop('animal')
    return counts
