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


@lru_cache(maxsize=1)
def noun_exceptions():
    with zipfile.ZipFile(DICTIONARY) as archive, archive.open('wordnet/noun.exc') as entries:
        pairs=[line.decode('utf-8').strip().split() for line in entries]
    return {row[0]:row[1:] for row in pairs if len(row)>1}


@lru_cache(maxsize=1024)
def species_noun(label):
    """Resolve number inflections only to an attested animal name."""
    if not label or not DICTIONARY.is_file():return label
    names,_=animal_names()
    if label in names:return label
    prefix,_,word=label.rpartition(' ')
    forms=set(noun_exceptions().get(word,()))
    for suffix,replacement in (('ies','y'),('es',''),('s','')):
        if word.endswith(suffix):
            forms.add(word[:-len(suffix)]+replacement)
    options={f'{prefix} {base}'.strip() for base in forms} & names.keys()
    return options.pop() if len(options)==1 else label


@lru_cache(maxsize=512)
def domestic_species(label):
    """Map named domestic breeds to the species counted in a scene inventory."""
    term=' '.join(re.findall('[a-z]+',label.casefold()))
    # Some modern breed names postdate WordNet 3.0.
    if term in ('shiba inu','shiba ken'):return 'dog'
    if not term or not DICTIONARY.is_file():return None
    names,parents=animal_names()
    level=set(names.get(term,()))
    visited=set()
    roots={s:species for lemma,species in [('domestic dog','dog'),('domestic cat','cat')]
           for s in names.get(lemma,())}
    while level:
        matches={roots[s] for s in level if s in roots}
        if len(matches)==1:return matches.pop()
        visited.update(level)
        level={p for s in level for p in parents.get(s,())}-visited
    return None


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
