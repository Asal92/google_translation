from google.cloud import translate_v2 as translate
import time

ENCODING = 'utf-8'

translator = translate.Client.from_service_account_json("../secret/high-comfort-368404-39708e023588.json")

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
assert False not in [True for tag in coner_tags if f"B-{tag[2:]}" and f"I-{tag[2:]}" in coner_tags], "Not every tag has a B and I tag"

UNK = 'unk'

def preproess_coner(sentence):
    new_string, original_string, tags_dict = [], [], {}
    if sentence == '':
        return None
    else:
        for w in sentence:
            if w[1]=="O":
                new_string.append(w[0])
                original_string.append(w[0])
            elif w[1] in coner_tags:
                # Replacing tags with unk since the tags were getting translated by Google cloud
                tags_dict[w[0]] = w[1]
                t = '[' + UNK + " " + w[0] + ']'
                new_string.append(t)
                original_string.append('[' + w[1] + " " + w[0] + ']')
        new_sentence = ' '.join(new_string)
        print("original sentence: ",' '.join(original_string))
    return new_sentence, tags_dict

def postprocess_coner(sentence, tags_dict, original):
    # this seems like an encoding issue - are we using utf-8?
    t = sentence
    t = t.replace("&quot;", "")
    t = t.replace("&#39;", "'")
    t = t.replace("&amp","&")
    t = t.replace("a&;s", "a&s")
    t = t.replace("&;", "&")
    t = t.replace("]-", " ")
    t = t.replace("],", " ")
    t = t.replace("].", " ")
    t = t.replace("[", " ")
    t = t.replace("]", " ")

    s = t.split()
    n, l = 0, len(s)

    for i in range(l):
        # Google cloud were translating words inside tags!!!
        # Google cloud translate words inside brackets with Upper case!
        if i<len(s):
            if s[i] == UNK:
                if n < len(tags_dict):
                    try:
                        s[i] = tags_dict[s[i+1].lower()]
                        s[i + 1] = s[i + 1].lower()
                        n += 1
                    except:
                        if i == len(s)-1: # Google cloud swapped the tokens inside the bracket
                            back_translate = translator.translate(s[i-1].lower(), source_language=target_language, target_language=source_language)
                            res = back_translate['translatedText'].lower()
                            if res in tags_dict.keys():
                                s[i] = tags_dict[res]
                                temp = s[i]
                                s[i] = s[i-1]
                                s[i-1] = temp
                                n += 1
                        else:
                            back_translate = translator.translate(s[i+1].lower(), source_language=target_language, target_language=source_language)
                            res = back_translate['translatedText'].lower()
                            if res in tags_dict.keys():
                                s[i] = tags_dict[res]
                                s[i + 1] = s[i + 1].lower()
                                n += 1
                            else:
                                # for now... plural s is missing after the translation
                                X, Y = s[i], s[i + 1]
                                s.remove(X)
                                s.remove(Y)
                                i -= 2
                else:
                    # unk at the end of thesentence. somehow translation is ruined again
                    if i == len(s)-1:
                        X = s[i]
                        s.remove(X)
                    else:
                        # Google cloud translated [X Y] twice!
                        X, Y = s[i], s[i+1]
                        s.remove(Y)
                        s.remove(X)
                        i -= 2
    if UNK in s: # watch out for leftovers
        s.remove(UNK)
    new_sentence = ' '.join(s)
    return new_sentence


def run(input_file_path, output_file_path):
    sentence = []
    print("Start the process.")
    with open(input_file_path, 'r', encoding=ENCODING) as input_file, open(output_file_path, 'w', encoding=ENCODING) as output_file:
        for line in input_file:
            line = line.strip()
            if line != '':
                if line[0] == '#':
                    output_file.write(f'\n{line}\n\n')
                    continue
                else:
                    line = line.split()
                    if len(line) == 2:
                        # LONDON: THIS DOESN'T SEEM TO BE HAPPENING BECAUSE THERE IS THE "_ _" PART OF THE DATASET
                        sentence.append(line)
            else:
                if sentence != []:
                    sentence_preprocessed, tags_dict = preproess_coner(sentence)
                    if sentence_preprocessed is None:
                        continue
                    
                    print("Here's what we're about to translate: ", sentence_preprocessed)
                    raise RuntimeError("Not processing - just testing")
                    results = translator.translate(sentence_preprocessed, source_language=source_language, target_language=target_language)
                    t = postprocess_coner(results['translatedText'], tags_dict, sentence)
                    print("translated sentence: ", t)
                    output_file.write(t + '\n')
                    time.sleep(0.2)
                    sentence = []

if __name__ == '__main__':
    import os
    # define source language and target language
    source_language = 'en'
    target_language = 'fr'
    # target_language = 'es'
    # target_language = 'nl'

    file_path = f'../data/{source_language}-train.conll'
    output_file_path = f'../output/{source_language}-to-{target_language}-train.conll'
    # make sure the output folder exists
    if not os.path.exists(os.path.dirname(output_file_path)):
        os.makedirs(os.path.dirname(output_file_path))
    run(file_path, output_file_path)
