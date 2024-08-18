import os
import random
import tensorflow
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential, Model
from keras.layers import LSTM, Dense, Embedding, Flatten, Input
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.optimizers import Adam
import xgboost as xgb
import joblib

model_file_lstm = 'model_lstm.keras'
model_file_rf = 'model_rf.joblib'
model_file_xgb = 'model_xgb.json'
model_file_ensemble = 'model_ensemble.joblib'
Exist_model = False
Batch_size = 32
Epochs = 100
Lim_percent_min = 0
Lim_percent_max = 101

file_paths = [
    'datasets/1398-.csv',
    'datasets/1399.csv',
    'datasets/1400.csv',
    'datasets/1401.csv',
    'datasets/1402-p1.csv',
    'datasets/1402-p2.csv',
]

file_remove_from_result = 'datasets/1401.csv'

test_file_path = 'datasets/1403.csv'

number_count = 105

def file_loader(paths: list):
    data_frames = []
    for file_path in paths:
        data_frames.append(pd.read_csv(file_path, header=None, delimiter=' '))
    combined_data = pd.concat(data_frames)
    return combined_data

def set_seed(seed_value=42):
    np.random.seed(seed_value)
    tensorflow.random.set_seed(seed_value)
    random.seed(seed_value)

def save_model(model, file_path):
    model.save(file_path)

def load_model(file_path):
    return tensorflow.keras.models.load_model(file_path)

class LSTMClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, input_dim, output_dim, max_len, epochs=100, batch_size=64):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.max_len = max_len
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = self.build_model()
        
    def build_model(self):
        model = Sequential()
        model.add(Embedding(input_dim=self.input_dim, output_dim=self.output_dim, input_length=self.max_len))
        model.add(LSTM(128))
        model.add(Flatten())
        model.add(Dense(4, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model
    
    def fit(self, X, y):
        y_binary = pd.get_dummies(y).values
        self.model.fit(X, y_binary, epochs=self.epochs, batch_size=self.batch_size)
    
    def predict_proba(self, X):
        return self.model.predict(X)
    
    def predict(self, X):
        return np.argmax(self.model.predict(X), axis=1)

def train_lstm(data, tokenizer, epochs: int = 100, batch_size: int = 64):
    X = data.iloc[:, 0].values
    Y = data.iloc[:, 1].values - 1  # Shift classes to start from 0
    X = X.astype(str)
    X_sequences = tokenizer.texts_to_sequences(X)
    X_sequences = np.array(X_sequences)
    
    lstm_model = LSTMClassifier(
        input_dim=len(tokenizer.word_index) + 1,
        output_dim=100,
        max_len=max(len(seq) for seq in X_sequences),
        epochs=epochs,
        batch_size=batch_size
    )
    
    if Exist_model and os.path.exists(model_file_lstm):
        lstm_model.model = load_model(model_file_lstm)
    else:
        lstm_model.fit(X_sequences, Y)
        save_model(lstm_model.model, model_file_lstm)

    return lstm_model

def train_rf(data, tokenizer):
    X = data.iloc[:, 0].values
    Y = data.iloc[:, 1].values - 1  # Shift classes to start from 0
    X = X.astype(str)
    X_sequences = tokenizer.texts_to_sequences(X)
    X_sequences = np.array(X_sequences)

    rf = RandomForestClassifier()
    
    if Exist_model and os.path.exists(model_file_rf):
        rf = joblib.load(model_file_rf)
    else:
        rf.fit(X_sequences, Y)
        joblib.dump(rf, model_file_rf)
    
    return rf

def train_xgb(data, tokenizer):
    X = data.iloc[:, 0].values
    Y = data.iloc[:, 1].values - 1  # Shift classes to start from 0
    X = X.astype(str)
    X_sequences = tokenizer.texts_to_sequences(X)
    X_sequences = np.array(X_sequences)

    xgb_model = xgb.XGBClassifier()
    
    if Exist_model and os.path.exists(model_file_xgb):
        xgb_model.load_model(model_file_xgb)
    else:
        xgb_model.fit(X_sequences, Y)
        xgb_model.save_model(model_file_xgb)
    
    return xgb_model

def ensemble_models(data, epochs=100, batch_size=64):
    tokenizer = Tokenizer()
    X = data.iloc[:, 0].astype(str).values
    tokenizer.fit_on_texts(X)
    
    lstm_model = train_lstm(data, tokenizer, epochs, batch_size)
    rf_model = train_rf(data, tokenizer)
    xgb_model = train_xgb(data, tokenizer)

    voting_clf = VotingClassifier(
        estimators=[('lstm', lstm_model), ('rf', rf_model), ('xgb', xgb_model)],
        voting='soft'  # Using soft voting for probabilities
    )

    X_sequences = tokenizer.texts_to_sequences(X)
    X_sequences = np.array(X_sequences)
    Y = data.iloc[:, 1].values - 1  # Shift classes to start from 0

    if Exist_model and os.path.exists(model_file_ensemble):
        voting_clf = joblib.load(model_file_ensemble)
    else:
        voting_clf.fit(X_sequences, Y)
        joblib.dump(voting_clf, model_file_ensemble)
    
    return voting_clf, lstm_model, rf_model, xgb_model

def combined_predictions_with_probabilities(outputs, top_k=1):
    combined_probs = []
    for probs in outputs:
        sorted_indices = np.argsort(probs)[::-1]
        top_indices = sorted_indices[:top_k]

        combined_prob = np.zeros_like(probs)

        for idx in top_indices:
            combined_prob[idx] = probs[idx]

        combined_prob = combined_prob / np.sum(combined_prob)
        combined_probs.append(combined_prob)

    return np.array(combined_probs)

def result(out_predict):
    questionnumber = 0
    out_dict = {}
    for line in out_predict:
        questionnumber += 1
        sor = sorted(line)
        largestnumber = sor[1]
        percent = round(largestnumber * 100, 1)
        option = list(line).index(largestnumber) + 1
        if percent <= Lim_percent_min or percent >= Lim_percent_max:
            percent = None
        out_dict[questionnumber] = [percent, option, line]
    return out_dict

def delete_file_from_predictions(filepath, resultout:dict):
    data = file_loader([filepath])
    for num in range(0, number_count):
        if resultout[num+1][1] == data[1][num]:
            resultout[num+1][0] = None
    return resultout

def random_delete_file_from_predictions(filepaths:list, resultout:dict):
    qus = 0
    filepaths.reverse()
    
    while True:
        for file in filepaths:
            data = file_loader([file])
            for _ in range(5):
                if qus == number_count:
                    return resultout
                if resultout[qus+1][1] == data[1][qus]:
                    resultout[qus+1][0] = None
                qus += 1        

def show(result: dict, epochs: int, batch_size: int):
    print('- In the name of God -'.center(78, '*'))
    print(f'Epochs: {epochs} -\nBatch_size: {batch_size} -')
    print('*' * 78)
    keys = sorted(result.keys())
    for questionnumber in keys:
        percentlist = [round(per * 100, 1) for per in result[questionnumber][2]]
        form = f'{questionnumber} - {percentlist}'
        charnummin = abs(70 - len(form))
        perform = str(result[questionnumber][0]).center(charnummin, '-')
        print(form, perform, f'[ {result[questionnumber][1]} ]\n')

def test(resultpredicted, test_file):
    testdict = {
        'Correct': 0,
        'Wrong': 0,
        'No answers': 0,
        'Sum': 0
    }
    comparisondata = file_loader([test_file])
    keys = sorted(resultpredicted.keys())
    for questionnumber in keys:
        if resultpredicted[questionnumber][0] is None:
            testdict['No answers'] += 1
        elif resultpredicted[questionnumber][1] == comparisondata.iloc[questionnumber - 1, 1]:
            testdict['Correct'] += 1
        else:
            testdict['Wrong'] += 1
    testdict['Sum'] = testdict['Correct'] + testdict['Wrong'] + testdict['No answers']
    return testdict



data = file_loader(file_paths)
set_seed(1)

ensemble_model, lstm_model, rf_model, xgb_model = ensemble_models(data, Epochs, Batch_size)

tokenizer = Tokenizer()
tokenizer.fit_on_texts(data.iloc[:, 0].astype(str).values)
input_for_next_year = [str(count) for count in range(1, 106)]
input_sequence = tokenizer.texts_to_sequences(input_for_next_year)
input_sequence = np.array(input_sequence)

predicted_output_ensemble = ensemble_model.predict_proba(input_sequence)
predicted_output_lstm = lstm_model.predict_proba(input_sequence)
predicted_output_rf = rf_model.predict_proba(input_sequence)
predicted_output_xgb = xgb_model.predict_proba(input_sequence)



print("Ensemble Model Results:")
last_result_ensemble = result(predicted_output_ensemble)
show(last_result_ensemble, Epochs, Batch_size)

print("LSTM Model Results:")
last_result_lstm = result(predicted_output_lstm)
show(last_result_lstm, Epochs, Batch_size)
test_outcome_lstm = test(last_result_lstm, test_file_path)
print(test_outcome_lstm)

print("Random Forest Model Results:")
last_result_rf = result(predicted_output_rf)
show(last_result_rf, Epochs, Batch_size)
test_outcome_rf = test(last_result_rf, test_file_path)
print(test_outcome_rf)

print("XGBoost Model Results:")
last_result_xgb = result(predicted_output_xgb)
show(last_result_xgb, Epochs, Batch_size)
test_outcome_xgb = test(last_result_xgb, test_file_path)
print(test_outcome_xgb)

# Evaluate ensemble model
test_outcome_ensemble = test(last_result_ensemble, test_file_path)
print("Test Outcome for Ensemble Model:")
print(test_outcome_ensemble)
