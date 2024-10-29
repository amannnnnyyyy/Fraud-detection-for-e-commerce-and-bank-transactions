import joblib
from sklearn.metrics import classification_report, accuracy_score

def model_training(models,X_train_data,X_test_data,y_train_data,y_test_data,dataType):
    for name, model in models.items():
        if name == 'RNN':
            model = Sequential()
            model.add(SimpleRNN(50, activation='relu', input_shape=(X_train_data.shape[1], 1)))
            model.add(Dense(1, activation='sigmoid'))
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            model.fit(X_train_data, y_train_data, epochs=10, batch_size=32)
            # Save the Keras RNN model
            model.save(f'saved_models/{name}.h5')
        elif name == 'LSTM':
            model = Sequential()
            model.add(LSTM(50, activation='relu', input_shape=(X_train_data.shape[1], 1)))
            model.add(Dense(1, activation='sigmoid'))
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            model.fit(X_train_data, y_train_data, epochs=10, batch_size=32)
            model.save(f'saved_models/{name}.h5')
        else:
            model.fit(X_train_data, y_train_data)
            joblib.dump(model, f'../models/{name}+{dataType}.pkl')

        # Make predictions
        if name in ['RNN', 'LSTM']:
            y_pred = model.predict(X_test_data).round()
        else:
            y_pred = model.predict(X_test_data)

        # Print classification report
        print(f"{name} (Credit Card Data):")
        print(f'Accuracy: {accuracy_score(y_test_data, y_pred)}')
        print(classification_report(y_test_data, y_pred))