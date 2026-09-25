import tensorflow as tf  # to specify and run computation graphs

def build_model(hidden_size, dropout_rate, lr):
    #define a basic sequential model
    model = tf.keras.Sequential()
    #rescale pixels from 0-255 to 0-1
    model.add(tf.keras.layers.Rescaling(scale=1./255))
    #flatten layer
    model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))

    model.add(tf.keras.layers.Dense(400, tf.nn.relu))
    model.add(tf.keras.layers.Dropout(0.2))

    model.add(tf.keras.layers.Dense(400, tf.nn.relu))
    model.add(tf.keras.layers.Dropout(0.2))

    model.add(tf.keras.layers.Dense(10, activation='softmax'))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False), metrics=['accuracy'])
    return model