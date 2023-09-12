import os
import pickle

from tensorflow.keras import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


def autoencoder_model(input_dims):
    # Część enkoder
    inputLayer = Input(shape=(input_dims,))
    encoder = Dense(128, activation="relu")(inputLayer)
    encoder = Dense(64, activation="relu")(encoder)
    encoder = Dense(32, activation="relu")(encoder)
    encoder = Dense(8, activation="relu")(encoder)

    # Część dekoder
    decoder = Dense(32, activation="relu")(encoder)
    decoder = Dense(64, activation="relu")(decoder)
    decoder = Dense(128, activation="relu")(decoder)
    decoder = Dense(input_dims, activation=None)(decoder)

    return Model(inputs=inputLayer, outputs=decoder)


# epochs = 30
# n_mels = 128
def train(training_dir, model_dir, n_mels, frame, lr, batch_size, epochs):
    # Ładowanie danych
    train_data_file = os.path.join(training_dir, 'autoenkoder_data.pkl')
    with open(train_data_file, 'rb') as f:
        train_data = pickle.load(f)

        # Przygotowanie modelu
    model = autoencoder_model(n_mels * frame)
    print(model.summary())

    # Kompilacja modelu
    model.compile(
        loss='mean_squared_error',
        optimizer=Adam(learning_rate=lr),
        metrics=['accuracy']
    )

    # Trenowanie modelu - te same dane używane do walidacji
    history = model.fit(
        train_data,
        train_data,
        batch_size=batch_size,
        validation_split=0.1,
        epochs=epochs,
        shuffle=True,
        verbose=2
    )

    # Zapis
    os.makedirs(model_dir, exist_ok=True)
    model.save(model_dir)
