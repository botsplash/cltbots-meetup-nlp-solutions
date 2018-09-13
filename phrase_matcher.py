#!/usr/bin/python
# -*- coding: iso-8859-15 -*-

import pandas as pd
import numpy as np

import spacy
from spacy.matcher import Matcher
from spacy.matcher import PhraseMatcher

import requests, json
from flask import Flask, request, Response

from articles_scraper.tools.nlp import get_text_tokens


def load_corpus(corpus_path):
    print("Loading QA corpus ...")
    corpus = pd.read_csv(corpus_path)
    # view sample review
    print( "Corpus Sample {}: {}".format(corpus["id"][24],
                                         corpus["question"][24]))
    return corpus

def build_phrase_matcher(corpus, nlp):
    print("Creating Phrase Matcher...")
    matcher = PhraseMatcher(nlp.vocab, max_length=10)

    def on_match(matcher, doc, id, matches):
        print("Matched!", matches)

    print("=====")
    print(" - Testing")
    matcher.add("1", on_match, nlp("Day 11 Ventilate ventilation Take Our 15 Day Spring"))
    print(" - Testing Done")

    print("=====")
    for index, row in corpus.iterrows():
        tokens = get_text_tokens(row["question"], 9)
        print(" - Adding: {}:{}".format(row["id"], tokens))
        patterns = [nlp(text) for text in tokens]
        matcher.add(str(row["id"]), on_match, *patterns)
    return matcher

def run_tests(matcher, corpus, nlp):
    print("=====")
    print("Sentence Tests...")
    sentences = [
        "avoid obsence images",
        "avoid repugnant situation",
        "avoid offensive images",
        "Documents needed for appraisal",
        "Tips to winterize home",
        "Best time to buy home for millenials",
        "Summer gardening tips for home",
        "Pressure Curb Appeal",
        "Pressure restrain Appeal",
        "Pressure wash Do Forget Curb Appeal",
        "Pressure wash Curb Appeal",
        "Curb appeal for summer",
        "Location sell To Homebuyers"
    ]
    for sentence in sentences:
        print("----")
        print("sentence: {}".format(sentence))
        doc = nlp(sentence)
        matches = matcher(doc)
        for match_id, start, end in matches:
            rule_id = nlp.vocab.strings[match_id]  # get the unicode ID, i.e. 'COLOR'
            span = doc[start : end]  # get the matched slice of the doc
            print(rule_id, span.text)
            found = corpus.loc[corpus["id"] == int(rule_id)]
            if not found.empty:
                print( "   - question: {}".format(found['question'].to_string()))
                print( "   - url: {}".format(found['url'].to_string()))
        print("-")
    print("=====")


print("Loading Corpus...")
corpus = load_corpus("examples/qa_corpus.csv")
print("Loading Spacy...")
nlp = spacy.load("en_core_web_lg")
print("Building Matcher...")
matcher = build_phrase_matcher(corpus, nlp)
print("Create Flask app...")
app = Flask(__name__)

@app.route('/match', methods = ['GET'])
def get_match():
    text = request.args.get('text')
    print("get_match: {}".format(text))
    if not text:
        return Response("Text not found", status=400,
                        mimetype='application/json')

    _resp = []

    doc = nlp(text)
    matches = matcher(doc)
    for match_id, start, end in matches:
        rule_id = nlp.vocab.strings[match_id]  # get the unicode ID, i.e. 'COLOR'
        span = doc[start : end]  # get the matched slice of the doc

        found = corpus.loc[corpus["id"] == int(rule_id)]
        question = found['question'].to_string() if not found.empty else None
        url = found['url'].to_string() if not found.empty else None

        _resp.append({
            "id": rule_id,
            "text": span.text,
            "question": question,
            "url": url,
        })
    print("-")

    data_send = json.dumps({
        'text': text,
        'result': _resp,
    })
    resp = Response(data_send, status=200, mimetype='application/json')
    resp.headers['Access-Control-Allow-Origin'] = "*"
    return resp


if __name__ == '__main__':
    print("Flask server starting...")
    app.debug = False
    app.run()
    print("Flask server stopped")
