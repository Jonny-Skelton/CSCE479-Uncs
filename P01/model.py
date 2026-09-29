import tensorflow as tf

def build_model(hidden_sizes, dropout_rate, lr):
    model = tf.keras.Sequential()
    model.add(tf.keras.Input(shape=(28, 28, 1)))
    model.add(tf.keras.layers.Rescaling(scale=1./255))
    model.add(tf.keras.layers.Flatten())

    for size in hidden_sizes:
        model.add(tf.keras.layers.Dense(size, tf.nn.relu))
        if dropout_rate > 0:
            model.add(tf.keras.layers.Dropout(dropout_rate))

    model.add(tf.keras.layers.Dense(10, activation='softmax'))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False), metrics=['accuracy'])
    return model
