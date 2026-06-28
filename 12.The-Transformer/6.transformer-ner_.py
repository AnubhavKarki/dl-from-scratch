# Named-Entity Recognition fine-tuning with DistilBERT (HuggingFace)
# Library: transformers, tensorflow, pandas, seqeval
# External data: ner.json, tokenizer/, model/
#
# Pipeline:
#   1. Load and clean resume JSON (mergeIntervals handles overlapping annotation spans)
#   2. Convert to (text, entities) pairs in spaCy format; trim whitespace from spans
#   3. Assign a tag string to every word; pad all tag sequences to MAX_LEN=512
#   4. Tokenize with DistilBertTokenizerFast — subtokens of the same word share the word's tag
#      (special tokens like [CLS]/[SEP] get label -100 so they are ignored by the loss)
#   5. Fine-tune TFDistilBertForTokenClassification with Adam, lr=1e-5
#
# Label alignment matters because BERT-style tokenizers split words into subword pieces.
# Word "Africa" might become ["Af", "##rica"] — both pieces must be tagged as Location.

import json
import re

import numpy as np
import pandas as pd
import tensorflow as tf


def load_and_clean(path="ner.json"):
    df = pd.read_json(path, lines=True).drop(["extras"], axis=1)
    df["content"] = df["content"].str.replace("\n", " ")
    return df


def merge_intervals(intervals):
    merged = []
    for hi in sorted(intervals, key=lambda t: t[0]):
        if not merged:
            merged.append(hi)
            continue
        lo = merged[-1]
        if hi[0] <= lo[1]:
            upper = max(lo[1], hi[1])
            merged[-1] = (lo[0], upper, lo[2]) if lo[2] is hi[2] else (
                (lo[0], upper, hi[2]) if lo[1] <= hi[1] else lo
            )
        else:
            merged.append(hi)
    return merged


def get_entities(df):
    entities = []
    for i in range(len(df)):
        entity = []
        for annot in df["annotation"][i]:
            try:
                ent = annot["label"][0]
                start = annot["points"][0]["start"]
                end = annot["points"][0]["end"] + 1
                entity.append((start, end, ent))
            except Exception:
                pass
        entities.append(merge_intervals(entity))
    return entities


def convert_dataturks_to_spacy(path):
    training_data = []
    with open(path) as f:
        for line in f:
            data = json.loads(line)
            text = data["content"].replace("\n", " ")
            entities = []
            for annot in (data["annotation"] or []):
                point = annot["points"][0]
                labels = annot["label"] if isinstance(annot["label"], list) else [annot["label"]]
                for label in labels:
                    ls = len(point["text"]) - len(point["text"].lstrip())
                    rs = len(point["text"]) - len(point["text"].rstrip())
                    entities.append((point["start"] + ls, point["end"] - rs + 1, label))
            training_data.append((text, {"entities": entities}))
    return training_data


def trim_entity_spans(data):
    invalid = re.compile(r"\s")
    cleaned = []
    for text, annotations in data:
        valid = []
        for start, end, label in annotations["entities"]:
            while start < len(text) and invalid.match(text[start]):
                start += 1
            while end > 1 and invalid.match(text[end - 1]):
                end -= 1
            valid.append([start, end, label])
        cleaned.append([text, {"entities": valid}])
    return cleaned


def tokenize_and_align_labels(tokenizer, examples, tags, max_length=512):
    tokenized = tokenizer(
        examples, truncation=True, is_split_into_words=False,
        padding="max_length", max_length=max_length,
    )
    labels = []
    for i, label in enumerate(tags):
        word_ids = tokenized.word_ids(batch_index=i)
        prev = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != prev:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(label[word_idx])
            prev = word_idx
        labels.append(label_ids)
    tokenized["labels"] = labels
    return tokenized


def train(ner_json_path="ner.json", tokenizer_path="tokenizer/",
          model_path="model/", epochs=10, batch_size=4):
    from tqdm.notebook import tqdm
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from transformers import DistilBertTokenizerFast, TFDistilBertForTokenClassification

    MAX_LEN = 512

    df = load_and_clean(ner_json_path)
    df["entities"] = get_entities(df)

    data = trim_entity_spans(convert_dataturks_to_spacy(ner_json_path))

    rows = []
    for text, annots in tqdm(data):
        words = text.split()
        word_tags = ["Empty"] * len(words)
        start = 0
        for wi, word in enumerate(words):
            end = start + len(word)
            for s, e, label in annots["entities"]:
                if start >= s and end <= e:
                    word_tags[wi] = label
                    break
            start = end + 1
        rows.append(word_tags)

    unique_tags = set(tag for row in rows for tag in row)
    tag2id = {tag: i for i, tag in enumerate(unique_tags)}

    tags_padded = pad_sequences(
        [[tag2id[t] for t in row] for row in rows],
        maxlen=MAX_LEN, value=tag2id["Empty"], padding="post", truncating="post",
    )

    tokenizer = DistilBertTokenizerFast.from_pretrained(tokenizer_path)
    tokenized = tokenize_and_align_labels(
        tokenizer, df["content"].values.tolist(), tags_padded
    )

    dataset = tf.data.Dataset.from_tensor_slices((
        tokenized["input_ids"], tokenized["labels"],
    ))

    model = TFDistilBertForTokenClassification.from_pretrained(
        model_path, num_labels=len(unique_tags)
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss=model.hf_compute_loss,
        metrics=["accuracy"],
    )
    model.fit(dataset.batch(batch_size), epochs=epochs, batch_size=batch_size)
    return model, tokenizer, tag2id


if __name__ == "__main__":
    print("Requires: ner.json, tokenizer/, model/ in the working directory.")
    print("Run: model, tokenizer, tag2id = train()")
