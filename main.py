import tensorflow.keras.layers as layers
import tensorflow.keras.losses as losses
import tensorflow.keras.optimizers as optimizers
import tensorflow.keras.models as models
import tensorflow.keras.utils as utils
import tensorflow.keras.callbacks as callbacks
from tensorflow.keras import Sequential
import tensorflow.data as data
from pickle import dump


train = utils.image_dataset_from_directory(
    'images',
    label_mode='categorical',
    labels='inferred',
    batch_size=64,
    image_size=(250, 250),
    seed=48,
    validation_split=0.3,
    subset='training',
)

test = utils.image_dataset_from_directory(
    'images',
    label_mode='categorical',
    labels='inferred',
    batch_size=64,
    image_size=(250, 250),
    seed=48,
    validation_split=0.3,
    subset='validation',
)

class_names = train.class_names
train = train.cache().prefetch(buffer_size=data.AUTOTUNE)
test = test.cache().prefetch(buffer_size=data.AUTOTUNE)

class Net():
    def __init__(self, image_size):
        self.model = models.Sequential()

        # dimensions: 250 x 250 x 3
        self.model.add(layers.Conv2D(32, 13, strides=3, input_shape=image_size, activation='relu'))
        # dimensions: 80 x 80 x 32
        self.model.add(layers.MaxPool2D(pool_size=2, strides=2))
        # dimensions: 40 x 40 x 32
        self.model.add(layers.Dropout(0.4))

        # dimensions: 40 x 40 x 32
        self.model.add(layers.Conv2D(64, 5, strides=1, activation='relu'))
        # dimensions: 36 x 36 x 64
        self.model.add(layers.MaxPool2D(pool_size=2, strides=2))
        # dimensions: 18 x 18 x 64
        self.model.add(layers.Dropout(0.4))

        # dimensions: 18 x 18 x 64
        self.model.add(layers.Conv2D(128, 3, strides=1, activation='relu'))
        # dimensions: 16 x 16 x 128
        self.model.add(layers.MaxPool2D(pool_size=2, strides=2))
        # dimensions: 8 x 8 x 128
        self.model.add(layers.Dropout(0.4))

        # dimensions: 8 x 8 x 128
        self.model.add(layers.Flatten())

        # dimensions: 8192
        self.model.add(layers.Dense(2048, activation='relu'))
        self.model.add(layers.Dropout(0.2))
        self.model.add(layers.Dense(512, activation='relu'))
        self.model.add(layers.Dropout(0.2))
        self.model.add(layers.Dense(128, activation='relu'))
        self.model.add(layers.Dropout(0.2))
        self.model.add(layers.Dense(5, activation='softmax'))

        self.loss = losses.CategoricalCrossentropy()
        # Adam is light on memory and efficient with larger datasets
        self.optimizer = optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=0.0000001)
        self.model.compile(
            loss=self.loss,
            optimizer=self.optimizer,
            metrics=['accuracy'],
        )

    def __str__(self):
        self.model.summary()
        return ''


net = Net((250, 250, 3))
print(net)

net.model.fit(
    train,
    batch_size=64,
    epochs=40,
    verbose=2,
    validation_data=test,
    validation_batch_size=64,
    callbacks=[
        callbacks.ModelCheckpoint(
            filepath='checkpoints/checkpoints_{epoch:02d}',
            verbose=1,
            monitor='val_accuracy',
            mode='max',
            save_best_only=True,
        ),
        # allow early stop to training when validation loss hasn't improved in 3 epochs
        callbacks.EarlyStopping(
            monitor='val_loss',
            mode='min',
            verbose=1,
            patience=3,
            restore_best_weights=True,
        ),
    ]
)

net.model.save('rice_model_save')
with open('rice_model_save/class_names.data', 'wb') as f:
    dump(class_names, f)
