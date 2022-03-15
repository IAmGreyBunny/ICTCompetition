import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import tensorflow as tf
from transformers import BertTokenizer
from transformers import TFBertModel
from model_config import create_model
from model_config import max_length_of_phrase,batch_size,num_of_classes,train_split,epochs,use_multiprocessing
import os

# Reads data from file into a panda dataframe
df = pd.read_csv(r"D:\ICT Competition\Dataset\train.csv", sep=',')

# View data info
df.info()                               #View Columns
print("dataset size: " + str(len(df)))  #View dataset size
print(df['label'].value_counts())       #View data in each label
# print(df.head())                      #View example of data

# Initialize bert tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

# Initialize input ids and attention mask
input_ids = np.zeros((len(df), max_length_of_phrase))
attn_masks = np.zeros((len(df), max_length_of_phrase))

# Function to generate training data
def generate_training_data(df, ids, masks, tokenizer):
    for i, text in tqdm(enumerate(df['text'])):
        print(f"tokenizing: {text}")
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
labels = np.zeros((len(df), num_of_classes))
labels[np.arange(len(df)), df['label'].values] = 1

dataset = tf.data.Dataset.from_tensor_slices((input_ids, attn_masks, labels))           # Creates dataset object

# Function to convert data into appropriate format for training
def SentimentDatasetMapFunction(input_ids, attn_masks, labels):
    return {
        'input_ids': input_ids,
        'attn_masks': attn_masks
    }, labels


dataset = dataset.map(SentimentDatasetMapFunction)

# Split data into batches of train and test
dataset = dataset.shuffle(5000).batch(batch_size, drop_remainder=True)     # batch size, drop any left out tensor
train_size = int((len(df)//batch_size)*train_split)                         # get size of training batch
valid_size = int(len(df)-train_size)
train_set = dataset.take(train_size)
validation_set = dataset.skip(train_size)
print("Train size each epochs: " + str(len(train_set))+"\nValidation size each epochs: "+str(len(validation_set)))

sentiment_model = create_model()

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
    epochs=epochs,
    callbacks=[early_stop_callback,model_checkpoint_callback],
    shuffle=True,
    use_multiprocessing=use_multiprocessing
)

sentiment_model.save('model/NLP Triaging')


