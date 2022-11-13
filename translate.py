import os
import re
import json
import warnings
from enum import Enum
from dataclasses import dataclass
from typing import List
from tqdm import tqdm
from google.cloud import translate_v2 as translate

ENCODING = 'utf-8'
# languages must be in Domain enum
SOURCE_LANGUAGE = "en"
TARGET_LANGUAGE = "fr"
INPUT_FOLDER = "data"
OUTPUT_FOLDER = "output"
INPUT_FILE_STRUCTURE = "{input_folder}/{source_language}-train.conll"
OUTPUT_FOLDER_STRUCTURE = "{output_folder}/{source_language}-{target_language}"
INPUT_FILE = INPUT_FILE_STRUCTURE.format(input_folder=INPUT_FOLDER, source_language=SOURCE_LANGUAGE)
OUTPUT_FOLDER = OUTPUT_FOLDER_STRUCTURE.format(output_folder=OUTPUT_FOLDER, source_language=SOURCE_LANGUAGE, target_language=TARGET_LANGUAGE)
JSON_FILE_NAME = "results.json"
JSON_FILE = f"{OUTPUT_FOLDER}/{JSON_FILE_NAME}"
UNTRANSLATED_CONLL_FILE_NAME = f"{TARGET_LANGUAGE}-orig-mulda.conll"
TRANSLATED_CONLL_FILE_NAME = f"{TARGET_LANGUAGE}-trans-mulda.conll"
UNTRANSLATED_CONLL_FILE = f"{OUTPUT_FOLDER}/{UNTRANSLATED_CONLL_FILE_NAME}"
TRANSLATED_CONLL_FILE = f"{OUTPUT_FOLDER}/{TRANSLATED_CONLL_FILE_NAME}"
RUN_GOOGLE_TRANSLATE = False
SECRET_JSON = "./secret/high-comfort-368404-39708e023588.json"
# how many full sentences to translate at once
# (each full sentence is translated once for each entity in the sentence plus once for the sentence as a whole)
BATCH_SIZE = 50

ID_VALUE_REGEX_KEY = "id_value"
DOMAIN_REGEX_KEY = "domain"
TOKEN_REGEX_KEY = "token"
TAG_TYPE_REGEX_KEY = "token_type"
TAG_REGEX_KEY = "tag"

START_BRACKET = '"'
END_BRACKET = '"'

TRANSLATED_TEXT_KEY = "translatedText"
INPUT_TEXT_KEY = "input"

class BracketsNotFoundWarning(UserWarning):
    '''Raised when brackets are not found in a string'''
    pass

class TagCategory(Enum):
    '''The tags are:
    Location (LOC) : Facility, OtherLOC, HumanSettlement, Station
    Creative Work (CW) : VisualWork, MusicalWork, WrittenWork, ArtWork, Software, OtherCW
    Group (GRP) : MusicalGRP, PublicCORP, PrivateCORP, OtherCORP, AerospaceManufacturer, SportsGRP, CarManufacturer, TechCORP, ORG
    Person (PER) : Scientist, Artist, Athlete, Politician, Cleric, SportsManager, OtherPER
    Product (PROD) : Clothing, Vehicle, Food, Drink, OtherPROD
    Medical (MED) : Medication/Vaccine, MedicalProcedure, AnatomicalStructure, Symptom, Disease
    (Plus an empty tag for untagged words)

    The CORP tags above are incorrect and should be "Corp" instead of "CORP"
    '''
    Empty = ''
    Facility = "Facility"
    OtherLOC = "OtherLOC"
    HumanSettlement = "HumanSettlement"
    Station = "Station"
    VisualWork = "VisualWork"
    MusicalWork = "MusicalWork"
    WrittenWork = "WrittenWork"
    ArtWork = "ArtWork"
    Software = "Software"
    OtherCW = "OtherCW"
    MusicalGRP = "MusicalGRP"
    PublicCorp = "PublicCorp"
    PrivateCorp = "PrivateCorp"
    OtherCorp = "OtherCorp"
    AerospaceManufacturer = "AerospaceManufacturer"
    SportsGRP = "SportsGRP"
    CarManufacturer = "CarManufacturer"
    TechCorp = "TechCorp"
    ORG = "ORG"
    Scientist = "Scientist"
    Artist = "Artist"
    Athlete = "Athlete"
    Politician = "Politician"
    Cleric = "Cleric"
    SportsManager = "SportsManager"
    OtherPER = "OtherPER"
    Clothing = "Clothing"
    Vehicle = "Vehicle"
    Food = "Food"
    Drink = "Drink"
    OtherPROD = "OtherPROD"
    MedicationVaccine = "Medication/Vaccine"
    MedicalProcedure = "MedicalProcedure"
    AnatomicalStructure = "AnatomicalStructure"
    Symptom = "Symptom"
    Disease = "Disease"

class TagType(Enum):
    '''Tag types are either B (beginning), I (inside), or O (outside)'''
    B = "B"
    I = "I"
    O = "O"

@dataclass
class Tag:
    '''A class to hold the tag information'''
    tag_type: TagType
    tag_category: TagCategory

    def __str__(self):
        return self.tag_format

    @property
    def tag_format(self):
        return f"{self.tag_type.value}{f'-{self.tag_category.value}' if self.tag_type != TagType.O else ''}"
    

class Domain(Enum):
    '''The domains are:
    BN-Bangla
    DE-German
    EN-English
    ES-Spanish
    FA-Farsi
    FR-French
    HI-Hindi
    IT-Italian
    PT-Portuguese
    SV-Swedish
    UK-Ukrainian
    ZH-Chinese
    '''
    BN = "bn"
    DE = "de"
    EN = "en"
    ES = "es"
    FA = "fa"
    FR = "fr"
    HI = "hi"
    IT = "it"
    PT = "pt"
    SV = "sv"
    UK = "uk"
    ZH = "zh"

@dataclass
class Word:
    '''A word is a token with a tag'''
    token: str
    tag: Tag

    def __str__(self):
        return f"{self.token} <{self.tag}>"

class Sentence:
    '''A sentence is essentially a list of words together with a list of indexes of the start of each entity'''

    def __init__(self, id_value=None, domain=None, words=None, entity_indexes=None):
        if words is None:
            words = []

        if entity_indexes is None:
            entity_indexes = []

        self.id_value = id_value
        self.domain = domain
        self.words = words
        self.entity_indexes = entity_indexes

    def __str__(self):
        return " ".join([str(word) for word in self.words])

    def add_word(self, word: Word):
        self.words.append(word)
        if word.tag.tag_type == TagType.B:
            self.entity_indexes.append(len(self.words) - 1)

    def get_bracketed_sentences(self) -> List[str]:
        '''For each entity in the sentence, return a string of the sentence with the entity bracketed'''
        bracketed_sentences = []
        for entity_index in self.entity_indexes:
            bracketed_sentence = ""
            bracketing = False
            for i, word in enumerate(self.words):
                bracketed_sentence += " "
                if i == entity_index:
                    bracketed_sentence += f"{START_BRACKET}{word.token}"
                    bracketing = True
                elif bracketing:
                    if word.tag.tag_type != TagType.I:
                        assert word.tag.tag_type == TagType.O or word.tag.tag_type == TagType.B
                        # remove the trailing space
                        bracketed_sentence = bracketed_sentence[:-1]
                        bracketed_sentence += f"{END_BRACKET} {word.token}"
                        bracketing = False
                    else:
                        bracketed_sentence += word.token
                else:
                    bracketed_sentence += word.token
            if bracketing:
                # close bracket if the entity is at the end of the sentence
                bracketed_sentence += END_BRACKET
            bracketed_sentence = bracketed_sentence.strip()
            if not check_brackets(bracketed_sentence):
                raise ValueError(f"Brackets were not done correctly in {bracketed_sentence}")
            bracketed_sentences.append(bracketed_sentence)
        assert len(bracketed_sentences) == len(self.entity_indexes)
        return bracketed_sentences

    def get_entity(self, entity_indexes_index: int) -> str:
        '''Return the entity at the given index'''
        entity_index = self.entity_indexes[entity_indexes_index]
        entity = ""
        for i, word in enumerate(self.words):
            if i == entity_index:
                entity += word.token
            elif i > entity_index:
                if word.tag.tag_type != TagType.I:
                    break
                else:
                    entity += f" {word.token}"
        return entity

    def get_entity_category(self, entity_indexes_index: int) -> TagCategory:
        '''Return the entity type at the given index'''
        entity_index = self.entity_indexes[entity_indexes_index]
        return self.words[entity_index].tag.tag_category

def check_brackets(sentence: str) -> bool:
    if START_BRACKET == END_BRACKET:
        return sentence.count(START_BRACKET) == 2
    else:
        return sentence.count(START_BRACKET) == sentence.count(END_BRACKET) == 1

def list_to_generator(input_list):
    '''Convert a list to a generator'''
    for item in input_list:
        yield item

def add_conll_word(file_desc, word: Word):
    '''Add a word to the (open) file'''
    file_desc.write(f"{word.token} _ _ {word.tag.tag_format}\n")

def add_conll_id_line(file_desc, id_value: str, domain: Domain):
    '''Add an ID to the (open) file'''
    # TODO this may need fixing
    file_desc.write(f"# id {id_value}\tdomain={domain.value}\n")

if RUN_GOOGLE_TRANSLATE:
    translator = translate.Client.from_service_account_json(SECRET_JSON)

input_file = open(INPUT_FILE, 'r', encoding=ENCODING)
sentences = []
sentence = Sentence()
# from https://stackoverflow.com/a/55188797/5049813
num_lines = sum(1 for line in input_file)
input_file.seek(0)
for line in tqdm(input_file, total=num_lines):
    line = line.strip()
    if not line:
        if sentence.words:
            sentences.append(sentence)
        sentence = Sentence()
    elif line.startswith("# id"):
        # use fullmatch to get the id and domain
        # the line should look like
        # # id bb81b9a7-e73d-4977-b6a8-0f7937123dfe   domain=en
        # (note that the domain is separated out by a tab)
        match = re.fullmatch(rf"# id (?P<{ID_VALUE_REGEX_KEY}>[a-zA-Z0-9-]+)\sdomain=(?P<{DOMAIN_REGEX_KEY}>[a-z]+)", line)
        if not match:
            raise ValueError(f"The id line is not formatted correctly: {line}")
        assert sentence.id_value is None, "The id value has already been set"
        sentence.id_value = match.group(ID_VALUE_REGEX_KEY)
        sentence.domain = Domain(match.group(DOMAIN_REGEX_KEY))
    else:
        # use fullmatch to get the token, tag, and tag type
        # the line should look like
        # `tongzhi _ _ B-OtherPER`
        # or
        # `founder _ _ O`
        # (spaces here are just spaces)
        match = re.fullmatch(rf"(?P<{TOKEN_REGEX_KEY}>\S+) _ _ (?P<{TAG_TYPE_REGEX_KEY}>[BIO])(-(?P<{TAG_REGEX_KEY}>[\S]+))?", line)
        if not match:
            raise ValueError(f"The word line is not formatted correctly: {line}")
        
        token = match.group(TOKEN_REGEX_KEY)
        tag_type = TagType(match.group(TAG_TYPE_REGEX_KEY))
        tag_category = TagCategory(match.group(TAG_REGEX_KEY)) if tag_type != TagType.O else TagCategory.Empty
        tag = Tag(tag_type, tag_category)
        word = Word(token, tag)
        sentence.add_word(word)

if sentence.words:
    sentences.append(sentence)

# we now have a list of sentences
# for each entity in each sentence, we want to put the entity in brackets and translate it
sentences_to_translate: List[List[str]] = [sentence.get_bracketed_sentences() for sentence in sentences]
if RUN_GOOGLE_TRANSLATE:
    results = []
    for i in tqdm(range(len(sentences_to_translate) // BATCH_SIZE + 1)):
        start = BATCH_SIZE * i
        end = min(BATCH_SIZE * (i + 1), len(sentences_to_translate))
        # see https://stackoverflow.com/questions/1198777/double-iteration-in-list-comprehension
        batch = [sentence for sentences in sentences_to_translate[start:end] for sentence in sentences]
        results.extend(translator.translate(batch, TARGET_LANGUAGE, 'text', SOURCE_LANGUAGE))
        break # testing
    
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    with open(JSON_FILE, 'w', encoding=ENCODING) as f:
        json.dump(results, f, indent=1)
        print(f"Results saved to {JSON_FILE}")

# we now have results in the json file
with open(JSON_FILE, 'r', encoding=ENCODING) as f:
    results = json.load(f)

# we want to put the results back into the CONLL format
results_generator = list_to_generator(results)
trans_file = open(TRANSLATED_CONLL_FILE, 'w', encoding=ENCODING)
orig_file = open(UNTRANSLATED_CONLL_FILE, 'w', encoding=ENCODING)
for sentence_index, sentence in enumerate(sentences):
    for entity_indexes_index, entity_index in enumerate(sentence.entity_indexes):
        try:
            result = next(results_generator)
        except StopIteration:
            raise RuntimeError(f"There wasn't one result for each entity. Was about to deal with entity {entity_indexes_index + 1} in the following sentence: {sentence}")
        translated_sentence = result[TRANSLATED_TEXT_KEY]
        entity_tag_category = sentence.get_entity_category(entity_indexes_index)
        print("translated sentence:", translated_sentence)

        if not check_brackets(translated_sentence):
            warnings.warn(f"Skipping! Could not find brackets in translated sentence: {translated_sentence}", BracketsNotFoundWarning)
            continue

        # separate off symbols that come before the starting bracket and after the ending bracket
        # for example, if there's a comma after the ending bracket
        start_bracket_location = translated_sentence.find(START_BRACKET)
        end_bracket_location = start_bracket_location + translated_sentence[start_bracket_location:].find(END_BRACKET)

        # add a space
        # since we're using split, extra spaces won't matter
        translated_sentence

        # add the id line
        example_id = f"{sentence.id_value}-{entity_indexes_index}"
        domain = Domain(TARGET_LANGUAGE)
        add_conll_id_line(trans_file, example_id, domain)
        add_conll_id_line(orig_file, example_id, domain)

        translated_words = translated_sentence.split()
        bracketing = False
        for translated_word in tqdm(translated_words, desc="words", leave=False):
            if translated_word.startswith(START_BRACKET):
                bracketing = True
                if translated_word.endswith(END_BRACKET):
                    bracketing = False
                    translated_word_without_brackets = translated_word[1:-1]
                else:
                    translated_word_without_brackets = translated_word[1:]

                orig_entity = sentence.get_entity(entity_indexes_index)
                orig_entity_words = orig_entity.split()

                # add this one translated word to the translated file
                word = Word(translated_word_without_brackets, Tag(TagType.B, entity_tag_category))
                add_conll_word(trans_file, word)

                # add the entire original word to the file
                for orig_entity_word_index, orig_entity_word in enumerate(orig_entity_words):
                    if orig_entity_word_index == 0:
                        entity_tag_type = TagType.B
                    else:
                        entity_tag_type = TagType.I
                    word = Word(orig_entity_word, Tag(entity_tag_type, entity_tag_category))
                    add_conll_word(orig_file, word)

            elif bracketing:
                if translated_word.endswith(END_BRACKET):
                    bracketing = False
                    translated_word_without_brackets = translated_word[:-1]
                word = Word(translated_word_without_brackets, Tag(TagType.I, entity_tag_category))
                add_conll_word(trans_file, word)
            else:
                word = Word(translated_word, Tag(TagType.O, TagCategory.Empty))
                add_conll_word(trans_file, word)
                add_conll_word(orig_file, word)
        trans_file.write("\n\n")
        orig_file.write("\n\n")
input_file.close()
trans_file.close()
orig_file.close()