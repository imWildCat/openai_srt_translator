#!/usr/bin/env python

import srt
import argparse
from openai import OpenAI, AzureOpenAI
import json
import os

DEPLOYMENT_ID = os.getenv("AZURE_DEPLOYMENT_ID")
AZURE_ENDPOINT = os.getenv("AZURE_ENDPOINT")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if DEPLOYMENT_ID and AZURE_ENDPOINT:
    client = AzureOpenAI(azure_deployment=DEPLOYMENT_ID, azure_endpoint=AZURE_ENDPOINT, api_key=OPENAI_API_KEY, api_version="2023-07-01-preview")
    print("using azure")
else:
    client = OpenAI(api_key=OPENAI_API_KEY)

BATCHSIZE = 50 # later i may use a token conter instead but this is simpler for now
LANG = "English"
MODEL = "gpt-3.5-turbo"
VERBOSE = False

def makeprompt():
    global prompt
    prompt = f"""You are a professional translator.
Translate the text below line by line into {LANG}, do not add any content on your own, and aside from translating, do not produce any other text, you will make the most accurate and authentic to the source translation possible.

these are subtitles, meaning each elements are related and in order, you can use this context to make a better translation.
you will reply with a json array that only contain the translation."""

def makebatch(chunk):
    return [x.content for x in chunk]

def translate_batch(batch):
    blen = len(batch)
    tbatch = []
    batch = json.dumps(batch, ensure_ascii=False)

    lendiff = 1
    while lendiff != 0: # TODO add max retry ?
        try:
            completion = client.chat.completions.create(model=MODEL,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": batch}
            ])
            tbatch = json.loads(completion.choices[0].message.content)
        except Exception as e:
            if VERBOSE:
                print(e)
            lendiff = 1
        else:
            lendiff = len(tbatch) - blen
    return tbatch

def translate_file(subs):
    total_batch = (len(subs) + BATCHSIZE - 1) // BATCHSIZE
    for i in range(0, len(subs), BATCHSIZE):
        print(f"batch {i//BATCHSIZE + 1} / {total_batch}")

        chunk = subs[i:i+BATCHSIZE]
        batch = makebatch(chunk)
        batch = translate_batch(batch)

        for j, n in enumerate(batch):
            chunk[j].content = n

def get_translated_filename(filepath):
    root, ext = os.path.splitext(os.path.basename(filepath))
    return f"{root}_{LANG}{ext}"

def main(files: str, language: str ="English", batch_size: int=50, model: str="gpt-3.5-turbo", verbose=False):
    global LANG, BATCHSIZE, MODEL, VERBOSE
    LANG = language
    BATCHSIZE = batch_size
    MODEL = model
    VERBOSE = verbose
    makeprompt()

    if not files:
        print("No files found matching the pattern.")
        return

    for filename in files:
        print(filename)
        sub = open(filename).read()
        subs = list(srt.parse(sub))

        translate_file(subs)
        output = srt.compose(subs)

        with open(get_translated_filename(filename),"w") as handle:
            handle.write(output)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Translate srt files")
    parser.add_argument("files", help="File pattern to match",nargs="+")
    parser.add_argument("-l", "--language", help="Specify the language", default="English", type=str)
    parser.add_argument("-b", "--batch_size", help="Specify the batch size", default=50, type=int)
    parser.add_argument("-m", "--model", help="openai's model to use", default="gpt-3.5-turbo", type=str)
    parser.add_argument("-v", "--verbose", help="display errors", action="store_true")

    args = parser.parse_args()

    main(args.files, args.language, args.batch_size, args.model, args.verbose)