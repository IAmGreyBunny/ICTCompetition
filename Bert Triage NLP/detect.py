import tensorflow as tf
from transformers import BertTokenizer
import numpy as np
from model_config import create_model
from model_config import label_map

model = create_model()
model.load_weights(r"D:\ICT Competition\model\chkpt\best.hdf5")

tokenizer = BertTokenizer.from_pretrained('bert-base-cased')


def prepare_data(input_text, tokenizer):
    token = tokenizer.encode_plus(
        input_text,
        max_length=256,
        truncation=True,
        padding='max_length',
        add_special_tokens=True,
        return_tensors='tf'
    )
    return {
        'input_ids': tf.cast(token.input_ids, tf.float64),
        'attn_masks': tf.cast(token.attention_mask, tf.float64)
    }


def make_prediction(model, processed_data, classes=label_map):
    probs = model.predict(processed_data)[0]
    return classes[np.argmax(probs)]


while True:
    input_text = input('Enter request here: ')
    processed_data = prepare_data(input_text, tokenizer)
    result = make_prediction(model, processed_data=processed_data)
    print(f"Predicted Emergency Level: {result}")
