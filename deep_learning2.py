# <--------- Assignment 1 -------------->

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import boston_housing
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense

df = pd.read_csv("HousingData.csv",na_values='NA')
df.head()

df.fillna(df.mean(numeric_only=True),inplace=True)
df.isna().sum()

x = df.loc[:, df.columns != 'MEDV']
y = df.loc[:, df.columns == 'MEDV']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

model = Sequential([
    Input(shape=(x_train.shape[1],)),
    Dense(128,activation='relu'),
    Dense(64,activation='relu'),
    Dense(1,activation='linear')
])

model.compile(optimizer='adam',loss='mse',metrics=['mae'])
history = model.fit(x_train_scaled,y_train,epochs=50,validation_split=0.2,verbose=0)

test_mse, test_mae = model.evaluate(x_test_scaled, y_test)
print('Mean squared error on test data: ', test_mse)
print('Mean absolute error on test data: ', test_mae)

predictions = model.predict(x_test_scaled)
for i in range(5):
    print(f"Predicted: {predictions[i][0]:.2f} Actual: {y_test.values[i]}")

plt.plot(history.history['mae'], label='Train MAE')
plt.plot(history.history['val_mae'], label='Validation MAE')
plt.xlabel('Epoch')
plt.ylabel('Mean Absolute Error')
plt.title('Training and Validation MAE')
plt.legend()
plt.show()


# <---------- Assignment 2 -------------->

import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Embedding, GlobalAveragePooling1D

df = pd.read_csv('IMDB Dataset.csv')
df.head()

reviews = df['review'].values
df['sentiment'] = df['sentiment'].map({'negative': 0, 'positive': 1})
sentiments = df['sentiment'].values

num_words = 10000
tokenizer = Tokenizer(num_words=num_words)
tokenizer.fit_on_texts(reviews)
sequences = tokenizer.texts_to_sequences(reviews)

max_len = 200
x_data = pad_sequences(sequences, maxlen=max_len)

x_train, x_test, y_train, y_test = train_test_split(x_data, sentiments, test_size=0.2)

model = Sequential([
    Input(shape=(max_len,)),
    Embedding(input_dim=num_words,output_dim=32),
    GlobalAveragePooling1D(),
    Dense(64,activation='relu'),
    Dense(1,activation='sigmoid')
])

model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
history = model.fit(x_train,y_train,epochs=10,batch_size=512,validation_data=(x_test,y_test),verbose=0)

loss, accuracy = model.evaluate(x_test,y_test)
print(f"Test Loss: {loss:.2f}")
print(f"Test Accuracy: {accuracy:.2f}")

pred_probs = model.predict(x_test)
predictions = (pred_probs>0.5).astype("int32")

for i in range(10):
    print(f"Review {i+1}: Predicted: {predictions[i][0]} Actual: {y_test[i]}")

plt.plot(history.history['accuracy'],label='Train Acc')
plt.plot(history.history['val_accuracy'],label='Val Acc')
plt.title('Training vs Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# <------------ Assignment 3 --------------------->

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical

train_df = pd.read_csv('./fashion_mnist/fashion-mnist_train.csv',dtype=np.uint8,nrows=10000)
test_df = pd.read_csv('./fashion_mnist/fashion-mnist_test.csv',dtype=np.uint8,nrows=2000)

x_train = (train_df.drop("label",axis=1).values/255.0).reshape(-1,28,28,1)
x_test = (test_df.drop("label",axis=1).values/255.0).reshape(-1,28,28,1)
y_train = to_categorical(train_df["label"],10)
y_test = to_categorical(test_df["label"],10)

train_df.head()

model = Sequential([
    Input(shape=(28, 28, 1)),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
history = model.fit(x_train,y_train,epochs=10,batch_size=512,validation_data=(x_test,y_test),verbose=1)

loss, accuracy = model.evaluate(x_test,y_test)
print(f"Test Loss: {loss:.2f}")
print(f"Test Accuracy: {accuracy:.2f}")

pred_probs = model.predict(x_test)
predictions = np.argmax(pred_probs, axis=1)
actuals = np.argmax(y_test, axis=1)

for i in range(10):
    print(f"Sample {i+1}: Predicted = {predictions[i]}, Actual = {actuals[i]}")

plt.plot(history.history['accuracy'],label='Train')
plt.plot(history.history['val_accuracy'],label='Test')
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

labels = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
for i in range(5):
    label_index = np.argmax(y_train[i])  # Convert one-hot to index
    print(labels[label_index])
    plt.imshow(x_train[i].reshape(28, 28), cmap='gray')
    plt.axis('off')
    plt.show()

# <------------ Assignment 4 --------------------->

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, SimpleRNN, Dense

data = pd.read_csv('goog.csv', parse_dates=['Date'], index_col='Date')
data = data[['Close']].dropna()  # Ensure no missing values

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

x = []
y = []
seq_len = 10

for i in range(seq_len, len(scaled_data)):
    x.append(scaled_data[i - seq_len:i])
    y.append(scaled_data[i])

x = np.array(x)
y = np.array(y)

split = int(0.8 * len(x))
x_train, x_test = x[:split], x[split:]
y_train, y_test = y[:split], y[split:]

print("Train shape:", x_train.shape, y_train.shape)
print("Test shape:", x_test.shape, y_test.shape)
print("Total rows in data:", len(scaled_data))

model = Sequential([
    Input(shape=(seq_len, 1)),
    SimpleRNN(50, activation='tanh'),
    Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')
history = model.fit(x_train, y_train,epochs=10,batch_size=32,validation_data=(x_test, y_test))

loss = model.evaluate(x_test,y_test);
print(f"Test loss: {loss:.2f}") 

predicted = model.predict(x_test)
predicted = scaler.inverse_transform(predicted)
actual = scaler.inverse_transform(y_test.reshape(-1, 1))

# Print first 10 predictions vs actual
print("Index\tActual\t\tPredicted")
for i in range(min(10, len(actual))):
    print(f"{i}\t{actual[i][0]:.2f}\t\t{predicted[i][0]:.2f}")

plt.figure(figsize=(10, 5))
plt.plot(actual, label='Actual')
plt.plot(predicted, label='Predicted')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()