import tensorflow as tf
from transformers import TFBertModel

# Hyperparameters
max_length_of_phrase = 256
num_of_classes = 6
batch_size = 2
train_split = 0.8
epochs = 1000
use_multiprocessing = False
label_map = [
    '1: Critically ill and requires resuscitation',
    '2: Major Emergency',
    '3: Minor Emergency',
    '4: Non Emergency',
    'Services',
    'Unknown'
]


def create_model():
    optim = tf.keras.optimizers.Adam(learning_rate=1e-5, decay=1e-6)
    loss_func = tf.keras.losses.CategoricalCrossentropy()
    acc = tf.keras.metrics.CategoricalAccuracy('accuracy')

    # Model
    bert_model = TFBertModel.from_pretrained('bert-base-cased')

    # Input layers
    input_ids = tf.keras.layers.Input(shape=(max_length_of_phrase,), name='input_ids', dtype='int32')
    attn_masks = tf.keras.layers.Input(shape=(max_length_of_phrase,), name='attn_masks', dtype='int32')

    bert_embds = bert_model.bert(input_ids, attention_mask=attn_masks)[1]
    intermediate_layer = tf.keras.layers.Dense(512, activation='relu', name='intermediate_layer')(bert_embds)
    output_layer = tf.keras.layers.Dense(num_of_classes, activation='softmax', name='output_layer')(intermediate_layer)

    # Compile model
    model = tf.keras.Model(inputs=[input_ids, attn_masks], outputs=output_layer)
    model.summary()
    model.compile(optimizer=optim, loss=loss_func, metrics=[acc])

    return model
