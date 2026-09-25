import json
import os
import numpy as np 
import tensorflow as tf
import tensorflow_datasets as tfds 

import util
from model import build_model

DATA_DIR = '../tensorflow-datasets/'
results_path = './results/'
save_path = results_path + 'saved_models/'
figure_path = results_path + 'figures/'
epoch_int = 10
max_epoch = 40
batch_size = 32
seed = 479

# the 2x2x2 grid
hidden_sizes = [[400, 400], [128]]
learning_rates = [1e-3, 1e-4]
dropouts = [0.0, 0.2]

os.makedirs(save_path, exist_ok=True)
os.makedirs(figure_path, exist_ok=True)

#partition data
training_set = tfds.load('fashion_mnist', split='train[:90%]', as_supervised=True, data_dir=DATA_DIR)
valid_set = tfds.load('fashion_mnist', split='train[-10%:]', as_supervised=True, data_dir=DATA_DIR)
testing_set = tfds.load('fashion_mnist', split='test', as_supervised=True, data_dir=DATA_DIR)

#set up data and batches
def prepare_data(data, shuffle=False):
    data = data.cache()
    if shuffle:
        data = data.shuffle(10000, seed=seed)
    data = data.batch(batch_size)
    data = data.prefetch(tf.data.AUTOTUNE)
    return data

training_set = prepare_data(training_set, shuffle=True)
valid_set = prepare_data(valid_set)
testing_set = prepare_data(testing_set)

f_labels = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat','Sandal','Shirt','Sneaker','Bag', 'Ankle boot']

results = []
histories = {}

for hidden in hidden_sizes:
    for lr in learning_rates:
        for dropout in dropouts:
            name = util.run_name(hidden, dropout, lr)
            tf.keras.utils.set_random_seed(seed)
            model = build_model(hidden, dropout, lr)

            early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)
            save_call = util.SaveModelEpoch(save_str=save_path + name + '_epoch', epoch_interval=epoch_int)

            record = model.fit(
                    training_set,
                    epochs=max_epoch,
                    batch_size=batch_size,
                    validation_data=valid_set,
                    callbacks=[early_stop, save_call],
                    verbose=2
                )
            histories[name] = record.history

            model.save(save_path + name + '.keras')

            val_loss, val_acc = model.evaluate(valid_set, verbose=0)
            test_loss, test_acc = model.evaluate(testing_set, verbose=0)
            results.append({
                'run': name,
                'hidden_sizes': '-'.join(str(h) for h in hidden),
                'learning_rate': lr,
                'dropout': dropout,
                'params': model.count_params(),
                'epochs_run': len(record.history['loss']),
                'best_epoch': int(np.argmin(record.history['val_loss'])) + 1,
                'train_loss': record.history['loss'][-1],
                'train_accuracy': record.history['accuracy'][-1],
                'val_loss': val_loss,
                'val_accuracy': val_acc,
                'test_loss': test_loss,
                'test_accuracy': test_acc,
            })
            print(name + ': val_acc=%.4f  test_acc=%.4f' % (val_acc, test_acc))

#the results table for the paper
results.sort(key=lambda r: r['val_accuracy'], reverse=True)
util.write_csv(results, results_path + 'results.csv')
with open(results_path + 'histories.json', 'w') as f:
    json.dump(histories, f, indent=2)

print('\n\nAll runs (ranked by validation accuracy):\n')
util.print_table(results)

best = results[0]
print('\nSelected on validation accuracy: ' + best['run'])
model = tf.keras.models.load_model(save_path + best['run'] + '.keras')

probs = model.predict(testing_set, verbose=0)
preds = np.argmax(probs, axis=1)
true_labels = np.concatenate([y for x, y in testing_set], axis=0)
images = np.concatenate([x for x, y in testing_set], axis=0)

n_test = len(true_labels)
test_acc = np.mean(preds == true_labels)
half, low, high = util.accuracy_ci(test_acc, n_test)
print('Test accuracy: %.4f +/- %.4f  (95%% CI [%.4f, %.4f], n = %d)' % (test_acc, half, low, high, n_test))

comparisons = []
for hidden in hidden_sizes:
    for lr in learning_rates:
        with_do = util.find_run(results, util.run_name(hidden, 0.2, lr))
        without_do = util.find_run(results, util.run_name(hidden, 0.0, lr))
        comparisons.append(util.error_difference_ci('dropout 0.2 vs none', with_do, without_do, n_test))
comparisons.append(util.error_difference_ci('best vs runner-up', results[0], results[1], n_test))

util.write_csv(comparisons, results_path + 'comparisons.csv')
print('\n95% CIs on the difference in test error (negative means model A is better,')
print('significant is True when the interval does not contain 0):\n')
util.print_table(comparisons)

# FIGURES
cm = util.confusion_matrix(true_labels, preds)
util.plot_confusion_matrix(cm, figure_path + 'confusion_matrix.png')
util.plot_histories(histories, figure_path + 'training_curves.png')

#random images plus where model was unsure
rng = np.random.default_rng(seed)
sample = rng.choice(n_test, size=24, replace=False)
errors = util.most_confident_errors(probs, preds, true_labels, n=24)
util.plot_prediction_grid(images, true_labels, preds, figure_path + 'predictions.png',
                          indices=sample, title='Random test predictions (' + best['run'] + ')')
util.plot_prediction_grid(images, true_labels, preds, figure_path + 'misclassified.png',
                          indices=errors, title='Most confident misclassifications')

#saliency
saliency = util.saliency_maps(model, images)
saliency_xi = util.saliency_maps(model, images, mode='grad_x_input')
sal_idx = np.concatenate([sample[:5], errors[:3]])
util.plot_saliency_grid(images, saliency, true_labels, preds, figure_path + 'saliency_examples.png', indices=sal_idx)
util.plot_saliency_grid(images, saliency_xi, true_labels, preds, figure_path + 'saliency_examples_gradxinput.png', indices=sal_idx)
util.plot_mean_saliency_per_class(images, saliency, true_labels, figure_path + 'saliency_per_class.png', saliency_xi=saliency_xi)

#per class accuracy
per_class = []
for c in range(10):
    others = np.where(np.arange(10) == c, -1, cm[c])
    per_class.append({
        'class': f_labels[c],
        'n': int(cm[c].sum()),
        'accuracy': cm[c, c] / cm[c].sum(),
        'most_confused_with': f_labels[int(np.argmax(others))],
    })
util.write_csv(per_class, results_path + 'per_class.csv')
print('\nPer-class test accuracy:\n')
util.print_table(per_class)
