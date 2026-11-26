"""Compare named animal varieties in a blind visual description using local WordNet."""
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
import re
import zipfile

DICTIONARY=Path(__file__).resolve().parents[1]/'models/lexicon/wordnet.zip'


@lru_cache(maxsize=1)
def animal_names():
    names=defaultdict(set)
    parents={}
    with zipfile.ZipFile(DICTIONARY) as archive, archive.open('wordnet/data.noun') as nouns:
        for line in nouns:
            if not line[:1].isdigit():continue
            fields=line.partition(b'|')[0].decode('utf-8').split()
            if fields[1]!='05':continue
            end=4+2*int(fields[3],16)
            for lemma in fields[4:end:2]:
                names[lemma.replace('_',' ').replace('-',' ').lower()].add(fields[0])
            links=fields[end+1:]
            parents[fields[0]]={links[i+1] for i in range(0,len(links),4) if links[i] in ('@','@i')}
    return names,parents


def named_animal_conflict(requirement,blind_description):
    if not DICTIONARY.is_file():return None
    if re.search(r'\b(?:possibly|likely|perhaps|resembles?|similar to|appears to be|either|or)\b',blind_description,re.I):
        return None
    if re.search(r'\b(?:without|not|no|or)\b',requirement,re.I):return None
    names,parents=animal_names()
    def terms(text):
        words=re.findall('[a-z]+',text.lower())
        found={}
        for i in range(len(words)):
            for length in range(min(6,len(words)-i),1,-1):
                term=' '.join(words[i:i+length])
                if len(names.get(term,()))==1:
                    found[term]=next(iter(names[term]))
                    break
        # "great horned owl" includes "horned owl"; retain the longer actual name.
        return {t:s for t,s in found.items() if not any(t!=u and (' '+t+' ') in (' '+u+' ') for u in found)}
    def family(synset):
        relatives={synset}
        level={synset}
        for _ in range(3):
            level={p for child in level for p in parents.get(child,())}-relatives
            relatives.update(level)
        return relatives
    wanted=terms(requirement)
    seen=terms(blind_description)
    if len(wanted)!=1 or len(seen)!=1:return None
    (target,a),=wanted.items()
    (observed,b),=seen.items()
    a_group,b_group=family(a),family(b)
    if a in b_group or b in a_group or not a_group.intersection(b_group):return None
    return {'requested':target,'observed':observed,'requested_synset':a,'observed_synset':b}
