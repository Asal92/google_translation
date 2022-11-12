import json
import os
from google.cloud import translate_v2 as translate
from tqdm import tqdm


tags_v2 = {'I-Artist': 1, 'B-Artist': 2, 'I-OtherPER': 3, 'I-VisualWork': 4, 'B-HumanSettlement': 5, 'I-MusicalWork': 6, 'I-Athlete': 7, 'B-OtherPER': 8, 'I-WrittenWork': 9, 'I-ORG': 10, 'B-Athlete': 11, 'I-Politician': 12, 'I-SportsGRP': 13, 'B-ORG': 14, 'I-MusicalGRP': 15, 'B-VisualWork': 16, 'B-MusicalWork': 17, 'B-Politician': 18, 'I-Facility': 19, 'B-MusicalGRP': 20, 'B-WrittenWork': 21, 'B-SportsGRP': 22, 'B-Facility': 23, 'B-OtherPROD': 24, 'I-HumanSettlement': 25, 'B-Software': 26, 'I-Scientist': 27, 'I-OtherLOC': 28, 'B-PublicCorp': 29, 'I-ArtWork': 30, 'I-PublicCorp': 31, 'I-OtherPROD': 32, 'I-SportsManager': 33, 'I-Cleric': 34, 'I-AerospaceManufacturer': 35, 'B-Disease': 36, 'B-Medication/Vaccine': 37, 'B-Scientist': 38, 'I-Software': 39, 'B-Food': 40, 'I-Station': 41, 'B-Vehicle': 42, 'B-SportsManager': 43, 'B-CarManufacturer': 44, 'B-Cleric': 45, 'B-AnatomicalStructure': 46, 'B-Drink': 47, 'B-Station': 48, 'I-CarManufacturer': 49, 'B-AerospaceManufacturer': 50, 'B-OtherLOC': 51, 'B-MedicalProcedure': 52, 'I-Vehicle': 53, 'B-Symptom': 54, 'B-Clothing': 55, 'B-ArtWork': 56, 'I-Symptom': 57, 'I-Disease': 58, 'I-PrivateCorp': 59, 'I-Drink': 60, 'B-PrivateCorp': 61, 'I-AnatomicalStructure': 62, 'I-Food': 63, 'I-MedicalProcedure': 64, 'I-Medication/Vaccine': 65, 'I-Clothing': 66}
coner_tags_brackets = [f'[{tag}' for tag in tags_v2.keys()]
coner_tags_dash = [f'{tag}-' for tag in tags_v2.keys()]


def preprocess(sentence):
    str = []
    for word, tag in sentence:
        if tag == "O":
            str.append(word)
        else:
            str.append(f'[{tag} {word}]')
    original_string = ' '.join(str)
    return original_string


def postprocess(sentence):
    print("I'm postprocessing this:")
    print(sentence)
    t = sentence
    # These should already be dealt with by Python via encodings
    # t = t.replace("\u201c", "\"")
    # t = t.replace("\u201d", "\"")
    # t = t.replace("\u201e", "\"")
    # t = t.replace("&quot;", "\"")
    # t = t.replace("&#39;", "'")
    # t = t.replace("&amp","&")
    t = t.replace(" DOCSTART ", "-DOCSTART-")
    for tag in tags_v2.keys():
        t = t.replace(f'{tag}-', f'{tag} ')
    s = t.split()

    start = 0
    while start < len(s) - 1:
        if s[start] in coner_tags_brackets:
            n = 0
            end = start + 1
            for j in range(end, len(s)):
                if ']' in s[j]: # London: I think this is the only line I changed here
                    end = j
                    break
            for j in range(end, start + 2, -1):
                s.insert(j, f'I-{s[start][3:]}')
                n += 1
            start = end + n
        else:
            start = start + 1

    new_sentence = ' '.join(s)
    new_sentence = new_sentence.replace("[", "").replace("]", "")

    return new_sentence


def unlinearize(sentence):
    unlin = ''
    i = 0
    s = sentence.split()
    while i < len(s):
        if s[i] in tags_v2:
            if i < len(s) - 1:
                unlin += f'{s[i + 1]} _ _ {s[i]}\n'
            i += 2
        else:
            unlin += f'{s[i]} _ _ O\n'
            i += 1
    return unlin


def main():
    source_lang = 'en'
    target_lang = 'fr'
    RUN_GOOGLE = False
    SECRET_JSON = "./secret/high-comfort-368404-39708e023588.json"
    translator = translate.Client.from_service_account_json(SECRET_JSON)

    ids = []
    sentences = []
    with open(f'data/{source_lang}-train.conll', 'r', encoding='utf-8') as input_file:
        sentence = []
        for line in input_file:
            line = line.strip()
            line = line.replace('_', '')
            if line != '':
                if line[0] == '#':
                    ids.append(line)
                else:
                    word_tag = line.split()
                    if len(word_tag) == 2:
                        sentence.append(word_tag)
            else:
                if sentence:
                    sentences.append(sentence)
                    sentence = []
    # sentences = list(sentences[:10])
    sources = [preprocess(sentence) for sentence in sentences]

    print("Here's what the sources look like:")
    print(sources[:5])
    sources = sources[:5] # make it smaller for testing

    if RUN_GOOGLE:
        results = []
        batch_size = 100
        for i in tqdm(range(len(sources) // batch_size + 1)):
            start = batch_size * i
            end = min(batch_size * (i + 1), len(sources))
            results.extend(translator.translate(list(sources[start:end]), target_lang, 'text', source_lang))

        assert len(results) == len(sources), f"len(results) = {len(results)}, len(sources) = {len(sources)}"

        os.makedirs(f'output/{source_lang}-{target_lang}', exist_ok=True)
        with open(f'output/{source_lang}-{target_lang}/results.txt', 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=1)

    with open(f'output/{source_lang}-{target_lang}/results.txt', 'r', encoding='utf-8') as f:
        results = json.load(f)

    translations = [(ids[i_result], postprocess(result['translatedText'])) for i_result, result in enumerate(results)]
    with open(f'output/{source_lang}-{target_lang}/trans.txt', 'w', encoding='utf-8') as f:
        json.dump(translations, f, indent=1)

    conll = [(id, unlinearize(trans)) for id, trans in translations]
    with open(f'output/{source_lang}-{target_lang}/{target_lang}-mulda.conll', 'w', encoding='utf-8') as f:
        for id, unlin in conll:
            f.write(id + '\n')
            f.write(unlin + '\n')


if __name__ == '__main__':
    main()
