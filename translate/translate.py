from google.cloud import translate_v2 as translate
import sys
import os
import time
from nltk import word_tokenize


translator = translate.Client.from_service_account_json("translation-367016-9ae157928fc9.json")


coner_tags =[
    # Location - LOC
    "B-Facility", "I-Facility", "B-OtherLOC", "I-OtherLOC", "B-HumanSettlement", "I-HumanSettlement", "B-Station", "I-Station",
    # Creative Work - CW
    "B-VisualWork", "I-VisualWork", "B-MusicalWork", "I-MusicalWork", "B-WrittenWork", "I-WrittenWork",
    "B-ArtWork", "I-ArtWork", "B-Software", "I-Software", "B-OtherCW", "I-OtherCW",
    # Group - GRP
    "B-MusicalGRP", "I-MusicalGRP", "B-PublicCORP", "I-PublicCORP", "B-PrivateCORP", "I-PrivateCORP",
    "B-OtherCORP", "I-OtherCORP", "B-AerospaceManufacturer", "I-AerospaceManufacturer", "B-SportsGRP", "I-SportsGRP",
    "B-CarManufacturer", "I-CarManufacturer", "B-TechCORP", "I-TechCORP", "B-ORG", "I-ORG",
    # Person - PER
    "B-Scientist", "I-Scientist", "B-Artist", "I-Artist", "B-Athlete", "I-Athlete", "B-Politician", "I-Politician",
    "B-Cleric", "I-Cleric", "B-SportsManager", "I-SportsManager", "B-OtherPER", "I-OtherPER",
    # Product - PROD
    "B-Clothing", "I-Clothing", "B-Vehicle", "I-Vehicle", "B-Food", "I-Food", "B-Drink", "I-Drink", "B-OtherPROD", "I-OtherPROD",
    # Medical - MED
    "B-Medication/Vaccine", "I-Medication/Vaccine", "B-MedicalProcedure", "I-MedicalProcedure",
    "B-AnatomicalStructure", "I-AnatomicalStructure", "B-Symptom", "I-Symptom", "B-Disease", "I-Disease",
]
UNK = 'unk'

def preproess_coner(sentence):
    new_string, original_string, tags = [], [], []
    if sentence == '':
        return None
    else:
        for w in sentence:
            if w[1]=="O":
                new_string.append(w[0])
                original_string.append(w[0])
            else:
                # Replacing tags with unk since the tags were getting translated by Google cloud
                tags.append(w[1])
                t = '[' + UNK + " " + w[0] + ']'
                new_string.append(t)
                original_string.append('[' + w[1] + " " + w[0] + ']')
        new_sentence = ' '.join(new_string)
        print("original sentence: ",' '.join(original_string))
    return new_sentence, tags

def postprocess_coner(sentence, tags):
    t = sentence
    t = t.replace("&quot;", "")
    t = t.replace("&#39;", "'")
    t = t.replace("&amp","&")
    t = t.replace("a&;s", "a&s")
    t = t.replace("[", "")
    t = t.replace("]", "")
    tag_n = 0

    s = t.split()
    for i in range(len(s)):
        # Google cloud were translating words inside tags!!!
        if s[i] == UNK:
            s[i] = tags[tag_n]
            tag_n += 1
        # Google cloud translate words inside brackets with Upper case!
        if s[i] in coner_tags:
            s[i+1] = s[i+1].lower()

    new_sentence = ' '.join(s)
    return new_sentence


def run(fpath, ofpath):
    sentence = []
    print("Start the process.")
    with open(fpath, 'r') as inf, open(ofpath, 'w') as of:
        for line in inf:
            line = line.strip()
            if line != '':
                if line[0] == '#':
                    of.write('\n')
                    of.write(line + '\n')
                    of.write('\n')
                    continue
                else:
                    line = line.split()
                    if len(line) == 2:
                        sentence.append(line)
            else:
                if sentence != []:
                    sentence_preprocessed, tags_list = preproess_coner(sentence)
                    if sentence_preprocessed is None:
                        continue
                    results = translator.translate(sentence_preprocessed, source_language=sl, target_language=tl)
                    t = postprocess_coner(results['translatedText'], tags_list)
                    print("translated sentence: ", t)
                    of.write(t + '\n')
                    time.sleep(0.2)
                    sentence = []

#define source language and target language
sl = 'en'
tl = 'de'
#tl = 'es'
#tl = 'nl'

fpath = 'en-mulda-train.txt'
ofpath = tl + '-' + fpath
run(fpath, ofpath)
