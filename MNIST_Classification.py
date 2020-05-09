import tensorflow as tf
import tensorflow_datasets as tfds

# --------------- Loading the dataset ----------------------
mnist_dataset , data_info = tfds.load(name='mnist', with_info = True, as_supervised = True)

# ------------------- Preprocessing ---------------

mnist_train , mnist_test = mnist_dataset['train'] , mnist_dataset['test']

# 10% of training set is for validation
num_validation_samples = 0.1 * data_info.splits['train'].num_examples
num_validation_samples = tf.cast(num_validation_samples, tf.int64)

num_test_samples = data_info.splits['test'].num_examples
num_test_samples = tf.cast(num_test_samples, tf.int64)

# Scaling the images' values
def scale_image(image, label):
    """ This function scales the image assuming it has greyscale values.
    0 -> 255 becomes 0 -> 1 """
    image = tf.cast(image, tf.float32)
    image /= 255.
    return image, label

scaled_train_valid_sets = mnist_train.map(scale_image)
scaled_test_sets = mnist_test.map(scale_image)

# Shuffling the dataset to make sure that it's randomized
BUFFER_SIZE = 10000
shuffled_train_valid_sets = scaled_train_valid_sets.shuffle(BUFFER_SIZE)
validation_data = shuffled_train_valid_sets.take(num_validation_samples)
train_data = shuffled_train_valid_sets.skip(num_validation_samples)

# Setting up the batches
BATCH_SIZE = 100
train_data = train_data.batch(BATCH_SIZE)
validation_data = validation_data.batch(num_validation_samples)
test_data = scaled_test_sets.batch(num_test_samples)

validation_inputs, validation_labels = next(iter(validation_data))

# ---------------- Outlining the Model -----------------------------
input_size = 784
output_size = 10
hidden_layer_size = 150

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape= (28,28,1)),
    tf.keras.layers.Dense(units = hidden_layer_size, activation='relu'),
    tf.keras.layers.Dense(units = hidden_layer_size, activation='relu'),
    tf.keras.layers.Dense(units = output_size, activation='softmax')
    ])

custom_optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0015)
model.compile(optimizer=custom_optimizer, loss = 'categorical_crossentropy', metrics = ['accuracy'])

# ------------------ Training ----------------------------------
NUM_EPOCHS = 6
model.fit(train_data, epochs = NUM_EPOCHS, verbose = 2, validation_data = validation_data)


# --------------------- Testing -------------------------------

test_loss , test_accuracy = model.evaluate(test_data, verbose=1)













