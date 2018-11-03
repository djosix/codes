import pickle, sys, os
import pandas as pd
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences

from config import defaults

model_path = sys.argv[1]
model_name = os.path.basename(model_path).rstrip('.save')

print('model_path', '=', model_path)

test_data = pd.read_csv('./data/test.tsv', delimiter='\t')
model = keras.models.load_model(model_path)
with open('./cache/dataset.pkl', 'rb') as f:
    test_x = pickle.load(f)['test_x']

pred_y = model.predict(pad_sequences(test_x, defaults['input_len'])).argmax(-1)

test_data['Sentiment'] = pred_y
test_data.to_csv('data/{}-pred.csv'.format(model_name), columns=['PhraseId', 'Sentiment'], index=False)
