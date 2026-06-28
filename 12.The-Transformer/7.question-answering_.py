# Extractive Question Answering with DistilBERT (HuggingFace)
# Library: transformers, tensorflow, torch, datasets, sklearn
# External data: data/ (bAbI dataset), tokenizer/, model/
#
# Task: given a context sentence and a question, predict the start and end token
# positions of the answer span inside the context.
#
# Data: bAbI two-sentence stories — two facts + one question with a single-word answer.
#   get_question_and_facts flattens the nested structure.
#   get_start_end_idx locates the answer string inside the joined context.
#   tokenize_align maps character-level span indices to subword token indices.
#
# TensorFlow training: custom loop with tf.GradientTape, two SparseCategoricalCrossentropy
#   losses (one for start position, one for end position), averaged 50/50.
#
# PyTorch training: HuggingFace Trainer with F1-score metrics on start and end positions.
#
# Both implementations use TFDistilBertForQuestionAnswering /
# DistilBertForQuestionAnswering loaded from a pre-trained checkpoint.

def load_babi(data_path="data/"):
    from datasets import load_from_disk
    return load_from_disk(data_path).flatten()


def get_question_and_facts(story):
    return {
        "question": story["story.text"][2],
        "sentences": " ".join([story["story.text"][0], story["story.text"][1]]),
        "answer": story["story.answer"][2],
    }


def get_start_end_idx(story):
    start = story["sentences"].find(story["answer"])
    return {"str_idx": start, "end_idx": start + len(story["answer"])}


def tokenize_align(example, tokenizer):
    encoding = tokenizer(
        example["sentences"], example["question"],
        truncation=True, padding=True, max_length=tokenizer.model_max_length,
    )
    start_pos = encoding.char_to_token(example["str_idx"])
    end_pos = encoding.char_to_token(example["end_idx"] - 1)
    if start_pos is None:
        start_pos = tokenizer.model_max_length
    if end_pos is None:
        end_pos = tokenizer.model_max_length
    return {
        "input_ids": encoding["input_ids"],
        "attention_mask": encoding["attention_mask"],
        "start_positions": start_pos,
        "end_positions": end_pos,
    }


def train_tensorflow(qa_dataset, model_path="model/tensorflow", epochs=3):
    import tensorflow as tf
    from transformers import TFDistilBertForQuestionAnswering

    train_ds = qa_dataset["train"]
    columns = ["input_ids", "attention_mask", "start_positions", "end_positions"]
    train_ds.set_format(type="tf", columns=columns)

    features = {x: train_ds[x] for x in ["input_ids", "attention_mask"]}
    labels = {
        "start_positions": tf.reshape(train_ds["start_positions"], [-1, 1]),
        "end_positions": tf.reshape(train_ds["end_positions"], [-1, 1]),
    }
    dataset = tf.data.Dataset.from_tensor_slices((features, labels)).batch(8)

    model = TFDistilBertForQuestionAnswering.from_pretrained(model_path, return_dict=False)
    opt = tf.keras.optimizers.Adam(learning_rate=3e-5)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    for epoch in range(epochs):
        print(f"epoch {epoch + 1}/{epochs}")
        for step, (x_batch, y_batch) in enumerate(dataset):
            with tf.GradientTape() as tape:
                start_scores, end_scores = model(x_batch)
                loss = 0.5 * (
                    loss_fn(y_batch["start_positions"], start_scores) +
                    loss_fn(y_batch["end_positions"], end_scores)
                )
            grads = tape.gradient(loss, model.trainable_weights)
            opt.apply_gradients(zip(grads, model.trainable_weights))
            if step % 20 == 0:
                print(f"  step {step:3d}  loss={float(loss):.4f}")
    return model


def train_pytorch(qa_dataset, model_path="model/pytorch", epochs=3):
    from sklearn.metrics import f1_score
    from transformers import DistilBertForQuestionAnswering, Trainer, TrainingArguments

    train_ds = qa_dataset["train"]
    test_ds = qa_dataset["test"]
    columns = ["input_ids", "attention_mask", "start_positions", "end_positions"]
    train_ds.set_format(type="pt", columns=columns)
    test_ds.set_format(type="pt", columns=columns)

    def compute_metrics(pred):
        return {
            "f1_start": f1_score(pred.label_ids[0], pred.predictions[0].argmax(-1), average="macro"),
            "f1_end": f1_score(pred.label_ids[1], pred.predictions[1].argmax(-1), average="macro"),
        }

    model = DistilBertForQuestionAnswering.from_pretrained(model_path)
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir="results", overwrite_output_dir=True,
            num_train_epochs=epochs, per_device_train_batch_size=8,
            per_device_eval_batch_size=8, warmup_steps=20,
            weight_decay=0.01, logging_dir=None, logging_steps=50,
        ),
        train_dataset=train_ds,
        eval_dataset=test_ds,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    return model, trainer


def answer_tensorflow(model, tokenizer, question, context):
    import tensorflow as tf
    inputs = tokenizer(context, question, return_tensors="tf",
                       truncation=True, padding="max_length",
                       max_length=tokenizer.model_max_length)
    start_logits, end_logits = model(inputs)
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"].numpy()[0])
    return " ".join(tokens[tf.math.argmax(start_logits, 1)[0]:tf.math.argmax(end_logits, 1)[0] + 1])


def answer_pytorch(model, tokenizer, question, context):
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    inputs = tokenizer(context, question, return_tensors="pt")
    outputs = model(inputs["input_ids"].to(device), attention_mask=inputs["attention_mask"].to(device))
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"].numpy()[0])
    return " ".join(tokens[torch.argmax(outputs[0], 1)[0]:torch.argmax(outputs[1], 1)[0] + 1])


if __name__ == "__main__":
    print("Requires: data/, tokenizer/, model/ directories.")
    print("Usage:")
    print("  from transformers import DistilBertTokenizerFast")
    print("  tokenizer = DistilBertTokenizerFast.from_pretrained('tokenizer/')")
    print("  qa = load_babi('data/').map(get_question_and_facts).map(get_start_end_idx)")
    print("  qa = qa.map(lambda ex: tokenize_align(ex, tokenizer))")
    print("  model = train_tensorflow(qa)  # or train_pytorch(qa)")
