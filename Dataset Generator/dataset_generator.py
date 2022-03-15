import os
from os.path import exists
import openai
import random
import csv
import time
import pandas as pd

api_key = "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"

openai.api_key = api_key


def generate_text(prompt):
    response = openai.Completion.create(
        engine="text-davinci-001",
        prompt=prompt,
        temperature=1,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    return response.choices[0].text


perspectives = ["patient", "patient's family member", "medical workers", "rescue workers"]

# Described by people with no medical knowledge
simple_conditions = [
    ["events that caused death", "medical incident that caused death", "severe injuries that caused death",
     "cause of death"],

    ["heart attack", "shortness of breath", "asthma", "unconsciousness", "strokes", "heavy bleeding", "convulsions",
     "severed limbs", "severe burns on face", "poisoning", "absent radial pulse", "bitten by poisonous animals",
     "large open wound", "fatal gunshot wounds", "fatal cut wounds"],

    ["chest pain with no heart attack", "dislocation of joints", "electrical burn", "chemical burn", "overdose",
     "limb fractures", "fractured arm", "fractured leg", "broken arm", "broken leg",
     "severe burns not on face", "major burns"],

    ["sprains", "insect stings or animal bites", "minor head injury", "foreign objects stuck in ear,nose or throat",
     "headaches", "minor burns", "minor bleeding", "small cuts", "minor injuries", "small burns", "minor cuts",
     "low-level of pain", "minor pain"],

    ["lower back pain", "acne", "coughing too much", "too much phlegm", "diarrhea", "constipation", "feverish",
     "minor illness", "infections", "sore throat", "mild symptoms", "mild illnesses", "mild symptoms covid",
     "past medical injuries", "past medical conditions", "chronic medical conditions"],

    ["thirst", "hunger", "feeling too hot", "feeling too cold", "need to go toilet", "need of clean up",
     "need for change of bedsheets", "room being too loud", "room being too bright", "room being too dark",
     "need for medicine"]
]

# Can be described by all people
all_conditions = [
    ["events that caused death", "medical incident that caused death", "severe injuries that caused death",
     "cause of death"],

    ["heart attack", "shortness of breath", "asthma", "unconsciousness", "strokes", "heavy bleeding", "convulsions",
     "severed limbs", "severe burns on face", "poisoning", "absent radial pulse", "bitten by poisonous animals",
     "large open wound", "fatal gunshot wounds", "fatal cut wounds"],

    ["chest pain with no heart attack", "dislocation of joints", "electrical burn", "chemical burn", "overdose",
     "limb fractures", "fractured arm", "fractured leg", "broken arm", "broken leg", "severe dehydration",
     "severe burns not on face", "major burns"],

    ["sprains", "insect stings or animal bites", "minor head injury", "foreign objects stuck in ear,nose or throat",
     "headaches", "minor burns", "minor bleeding", "small cuts", "minor injuries", "small burns", "minor cuts",
     "low-level of pain", "minor pain"],

    ["lower back pain", "acne", "coughing too much", "too much phlegm", "diarrhea", "constipation", "feverish",
     "minor illness", "infections", "sore throat", "mild symptoms", "mild illnesses", "mild symptoms covid",
     "past medical injuries", "past medical conditions", "chronic medical conditions"],

    ["thirst", "hunger", "feeling too hot", "feeling too cold", "need to go toilet", "need of clean up",
     "need for change of bedsheets", "room being too loud", "room being too bright", "room being too dark",
     "need for medicine"]
]

random_topics = ["football", "politics", "hobbies", "fashion", "tourism", "scenery", "family", "travel", "social media",
                 "goals", "opinions", "history", "geography", "culture", "entertainment", "technology"]


def prompt_construction(pacs_category):
    if pacs_category >= 6:
        prompt = "a sentence from a conversation about " + random.choice(random_topics)
    elif pacs_category == 0:
        prompt = random.choice(perspectives) + random.choice(all_conditions[pacs_category])
    elif pacs_category > 3:  # Only non-emergency workers
        perspective = random.choice(perspectives[:2])
        patient_condition = random.choice(simple_conditions[pacs_category])
        if perspective is perspectives[0]:  # first person
            prompt = perspective + " describing current " + patient_condition
        else:  # third person
            prompt = perspective + " describing patient current " + patient_condition
    else:
        perspective = random.choice(perspectives)
        if perspective is perspectives[0]:  # first person
            patient_condition = random.choice(simple_conditions[pacs_category])
            prompt = perspective + " describing current " + patient_condition
        elif perspective == perspectives[2] or perspective == perspectives[3]:  # third person
            patient_condition = random.choice(all_conditions[pacs_category])
            prompt = perspective + " describing patient current " + patient_condition
        else:  # third person
            patient_condition = random.choice(simple_conditions[pacs_category])
            prompt = perspective + " describing patient current " + patient_condition
    return prompt


if exists("train.csv"):
    print("train.csv already exist")
else:
    with open('train.csv', 'w', newline='', encoding='UTF8') as f:
        header = ['text', 'label']
        writer = csv.writer(f)
        writer.writerow(header)
    print("train.csv created")

for i in range(0, 10000):
    for pacs_category in range(0, 7):
        try:
            prompt = prompt_construction(pacs_category)
            generated_sentence = generate_text(prompt).strip()
            generated_sentence = generated_sentence.replace("\n", "")
            generated_sentence = generated_sentence.replace('"', '')
            if generated_sentence:  # make sure string is not empty
                with open('train.csv', 'a', newline='', encoding='UTF8') as f:
                    data_entry = [generated_sentence, pacs_category]
                    writer = csv.writer(f)
                    writer.writerow(data_entry)
                print("Number of iterations: " + str(i*7+pacs_category))
            time.sleep(1)
        except Exception:
            continue

