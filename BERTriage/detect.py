import tensorflow as tf
from transformers import BertTokenizer
import numpy as np
from BERTriage.model_config import create_model
from BERTriage.model_config import label_map

physical_devices = tf.config.list_physical_devices('GPU')
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)

def load_model(model_path):
    model = create_model()
    model.load_weights(model_path)
    return model


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


def make_prediction(model, data, classes=label_map):
    processed_data = prepare_data(data, BertTokenizer.from_pretrained('bert-base-cased'))
    probs = model.predict(processed_data)[0]
    return classes[np.argmax(probs)]
