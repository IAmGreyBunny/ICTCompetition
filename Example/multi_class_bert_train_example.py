import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import tensorflow as tf
from transformers import BertTokenizer
import os

# Hyperparameters
max_length_of_phrase = 256
num_of_classes = 5
batch_size = 2
train_split = 0.8
optim = tf.keras.optimizers.Adam(learning_rate=1e-5, decay=1e-6)
loss_func = tf.keras.losses.CategoricalCrossentropy()
acc = tf.keras.metrics.CategoricalAccuracy('accuracy')

# Reads data from file into a panda dataframe
df = pd.read_csv("../train.tsv", sep='\t')

# View data info
# df.info()                                   #View Columns
# print(df['sentiments'].value_counts())      #View data in each label
# print(df.head())                            #View example of data

# Initialize bert tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

# Initialize input ids and attention mask
input_ids = np.zeros((len(df), max_length_of_phrase))
attn_masks = np.zeros((len(df), max_length_of_phrase))

# Function to generate training data
def generate_training_data(df, ids, masks, tokenizer):
    for i, text in tqdm(enumerate(df['Phrase'])):
        tokenized_text = tokenizer.encode_plus(
            text,
            max_length=max_length_of_phrase,
            truncation=True,
            padding='max_length',
            add_special_tokens=True,
            return_tensors='tf'
        )
        ids[i, :] = tokenized_text.input_ids
        masks[i, :] = tokenized_text.attention_mask
    return ids, masks


input_ids, attn_masks = generate_training_data(df, input_ids, attn_masks, tokenizer)

# Encode the label with int, this tells the model what label for each phrase
labels = np.zeros((len(df), 5))
labels[np.arange(len(df)), df['Sentiment'].values] = 1

dataset = tf.data.Dataset.from_tensor_slices((input_ids, attn_masks, labels))           # Creates dataset object

# Function to convert data into appropriate format for training
def SentimentDatasetMapFunction(input_ids, attn_masks, labels):
    return {
        'input_ids': input_ids,
        'attn_masks': attn_masks
    }, labels


dataset = dataset.map(SentimentDatasetMapFunction)

# Split data into batches of train and test
dataset = dataset.shuffle(1000).batch(batch_size, drop_remainder=True)     # batch size, drop any left out tensor
train_size = int((len(df)//batch_size)*train_split)                         # get size of training batch
train_set = dataset.take(train_size)
validation_set = dataset.skip(train_size)

# Loads pre-trained Bert Model
from transformers import TFBertModel
bert_model = TFBertModel.from_pretrained('bert-base-cased')

# Input layers
input_ids = tf.keras.layers.Input(shape=(max_length_of_phrase,),name='input_ids',dtype='int32')
attn_masks = tf.keras.layers.Input(shape=(max_length_of_phrase,),name='attn_masks',dtype='int32')

bert_embds = bert_model.bert(input_ids,attention_mask=attn_masks)[1]
intermediate_layer = tf.keras.layers.Dense(512,activation='relu',name='intermediate_layer')(bert_embds)
output_layer = tf.keras.layers.Dense(num_of_classes,activation='softmax', name ='output_layer')(intermediate_layer)

# Compile model
sentiment_model = tf.keras.Model(inputs=[input_ids,attn_masks],outputs=output_layer)
sentiment_model.summary()
sentiment_model.compile(optimizer=optim, loss=loss_func, metrics=[acc])


# Model Training
checkpoint_filepath = os.path.join('model/chkpt/epoch{epoch:02d}-acc{val_accuracy:.2f}.hdf5')

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=False,
    monitor='val_accuracy',
    mode='max',
    save_freq='epoch',
    save_best_only=True)

early_stop_callback = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, mode="min")

hist = sentiment_model.fit(
    train_set,
    validation_data=validation_set,
    epochs=100,
    callbacks=[early_stop_callback,model_checkpoint_callback],
    shuffle=True,
    steps_per_epoch=2500,
    validation_steps=2500
)

sentiment_model.save('sentiment_model')


# TESTING MODEL
# sentiment_model = tf.keras.models.load_model('sentiment_model')
#
# tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
#
# def prepare_data(input_text, tokenizer):
#     token = tokenizer.encode_plus(
#         input_text,
#         max_length=256,
#         truncation=True,
#         padding='max_length',
#         add_special_tokens=True,
#         return_tensors='tf'
#     )
#     return {
#         'input_ids': tf.cast(token.input_ids, tf.float64),
#         'attention_mask': tf.cast(token.attention_mask, tf.float64)
#     }
#
# def make_prediction(model, processed_data, classes=['Negative', 'A bit negative', 'Neutral', 'A bit positive', 'Positive']):
#     probs = model.predict(processed_data)[0]
#     return classes[np.argmax(probs)]
#
#
# input_text = input('Enter movie review here: ')
# processed_data = prepare_data(input_text, tokenizer)
# result = make_prediction(sentiment_model, processed_data=processed_data)
# print(f"Predicted Sentiment: {result}")


