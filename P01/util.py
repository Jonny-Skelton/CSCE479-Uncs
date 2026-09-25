import csv

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf 

CLASS_NAMES = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


class SaveModelEpoch(tf.keras.callbacks.Callback):
  def __init__(self,save_str, epoch_interval):
      super().__init__()
      self.save_str=save_str
      self.epoch_interval=epoch_interval
      self.count=0

  def on_epoch_end(self, epoch, logs=None):
      if epoch % self.epoch_interval == 0:
          save_str_count=self.save_str+str(self.count)+'.keras'
          self.model.save(save_str_count)
          self.count+=1


def run_name(hidden_sizes, dropout_rate, lr):
    arch = '-'.join(str(h) for h in hidden_sizes)
    return 'h%s_lr%.0e_do%s' % (arch, lr, dropout_rate)


def find_run(results, name):
    for row in results:
        if row['run'] == name:
            return row


def accuracy_ci(accuracy, n):
    half = 1.96 * np.sqrt(accuracy * (1 - accuracy) / n)
    return half, accuracy - half, accuracy + half


def error_difference_ci(label, row_a, row_b, n):
    err_a = 1 - row_a['test_accuracy']
    err_b = 1 - row_b['test_accuracy']
    diff = err_a - err_b
    half = 1.96 * np.sqrt(err_a * (1 - err_a) / n + err_b * (1 - err_b) / n)
    return {'comparison': label,
            'model_a': row_a['run'],
            'model_b': row_b['run'],
            'error_a': err_a,
            'error_b': err_b,
            'difference': diff,
            'ci_low': diff - half,
            'ci_high': diff + half,
            'significant': abs(diff) > half}


def confusion_matrix(trues, preds):
    cm = np.zeros((10, 10), dtype=int)
    for t, p in zip(trues, preds):
        cm[t, p] += 1
    return cm


def write_csv(rows, path):
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def print_table(rows):
    columns = list(rows[0].keys())
    cells = []
    for row in rows:
        cells.append(['%.4f' % row[c] if isinstance(row[c], float) else str(row[c])
                      for c in columns])
    widths = [max([len(c)] + [len(cell[i]) for cell in cells])
              for i, c in enumerate(columns)]
    header = '  '.join(c.ljust(w) for c, w in zip(columns, widths))
    print(header)
    print('-' * len(header))
    for cell in cells:
        print('  '.join(c.ljust(w) for c, w in zip(cell, widths)))


#FIGURES
def _squeeze(image):
    return np.squeeze(image, axis=-1) if image.ndim == 3 else image


def plot_confusion_matrix(cm, path, class_names=CLASS_NAMES, normalize=True):
    data = cm.astype(float)
    if normalize:
        data = data / np.maximum(data.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(8.5, 7.5))
    im = ax.imshow(data, cmap='Blues', vmin=0, vmax=data.max())
    ax.set_xticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.set_yticks(range(len(class_names)))
    ax.set_yticklabels(class_names)
    ax.set_xlabel('Predicted label')
    ax.set_ylabel('True label')
    ax.set_title('Confusion matrix' + (' (row-normalized)' if normalize else ''))
    threshold = data.max() / 2.0
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if normalize:
                text = '{:.0%}'.format(data[i, j]) if data[i, j] >= 0.005 else ''
            else:
                text = str(int(cm[i, j]))
            ax.text(j, i, text, ha='center', va='center', fontsize=7,
                    color='white' if data[i, j] > threshold else 'black')
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_histories(histories, path):
    """Training / validation curves for every run in the grid."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    cmap = plt.get_cmap('tab10')
    for idx, (name, hist) in enumerate(histories.items()):
        color = cmap(idx % 10)
        epochs = range(1, len(hist['loss']) + 1)
        axes[0].plot(epochs, hist['loss'], color=color, alpha=0.35)
        axes[0].plot(epochs, hist['val_loss'], color=color, label=name)
        axes[1].plot(epochs, hist['accuracy'], color=color, alpha=0.35)
        axes[1].plot(epochs, hist['val_accuracy'], color=color, label=name)
    axes[0].set_title('Loss (faded = train, solid = validation)')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Cross-entropy')
    axes[1].set_title('Accuracy (faded = train, solid = validation)')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend(fontsize=7, loc='lower right')
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_prediction_grid(images, trues, preds, path, class_names=CLASS_NAMES,
                         indices=None, n=24, ncols=6, title='Test predictions'):
    """Grid of test images titled with predicted and true class; wrong = red."""
    if indices is None:
        indices = np.arange(min(n, len(images)))
    indices = np.asarray(indices)[:n]
    nrows = int(np.ceil(len(indices) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.0 * ncols, 2.3 * nrows),
                             squeeze=False)
    flat = np.ravel(axes)
    for ax, idx in zip(flat, indices):
        ax.imshow(_squeeze(images[idx]), cmap='gray')
        correct = preds[idx] == trues[idx]
        ax.set_title('pred: {}\ntrue: {}'.format(class_names[preds[idx]],
                                                 class_names[trues[idx]]),
                     fontsize=8, color='green' if correct else 'red')
        ax.axis('off')
    for ax in flat[len(indices):]:
        ax.axis('off')
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def most_confident_errors(probs, preds, trues, n=24):
    """Indices of the misclassifications the model was most confident about."""
    wrong = np.flatnonzero(preds != trues)
    if len(wrong) == 0:
        return wrong
    confidence = probs[wrong, preds[wrong]]
    return wrong[np.argsort(-confidence)][:n]


#  saliency

def saliency_maps(model, images, batch_size=256, mode='grad'):
    maps = []
    for start in range(0, len(images), batch_size):
        x = tf.cast(images[start:start + batch_size], tf.float32)
        with tf.GradientTape() as tape:
            tape.watch(x)
            probs = model(x, training=False)
            pred = tf.argmax(probs, axis=1)
            score = tf.gather(probs, pred, batch_dims=1)
        grad = tape.gradient(score, x)
        if mode == 'grad_x_input':
            grad = grad * x
        maps.append(tf.abs(grad).numpy())
    return np.concatenate(maps, axis=0)


def _prep_saliency(sal, smooth=True):
    s = _squeeze(sal).astype(np.float32)
    if smooth:
        padded = np.pad(s, 1, mode='edge')
        s = sum(padded[i:i + s.shape[0], j:j + s.shape[1]]
                for i in range(3) for j in range(3)) / 9.0
    return np.clip(s / max(np.percentile(s, 99), 1e-12), 0, 1)


def plot_saliency_grid(images, saliency, trues, preds, path,
                       class_names=CLASS_NAMES, indices=None, n=8):
    if indices is None:
        indices = np.arange(min(n, len(images)))
    indices = np.asarray(indices)[:n]
    fig, axes = plt.subplots(3, len(indices), figsize=(1.7 * len(indices), 5.6),
                             squeeze=False)
    for col, idx in enumerate(indices):
        img = _squeeze(images[idx])
        sal = _prep_saliency(saliency[idx])
        correct = preds[idx] == trues[idx]
        axes[0, col].imshow(img, cmap='gray')
        axes[0, col].set_title('pred: {}\ntrue: {}'.format(
            class_names[preds[idx]], class_names[trues[idx]]),
            fontsize=8, color='green' if correct else 'red')
        axes[1, col].imshow(sal, cmap='inferno')
        axes[2, col].imshow(img, cmap='gray')
        axes[2, col].imshow(sal, cmap='inferno', alpha=0.55)
        for row in range(3):
            axes[row, col].axis('off')
    fig.suptitle('Input (top) / saliency (middle) / overlay (bottom)')
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_mean_saliency_per_class(images, saliency, trues, path,
                                 saliency_xi=None, class_names=CLASS_NAMES,
                                 max_per_class=200):
    n_classes = len(class_names)
    rows = ['Mean image', '|gradient|'] + (['|gradient x input|'] if saliency_xi is not None else [])
    fig, axes = plt.subplots(len(rows), n_classes,
                             figsize=(1.5 * n_classes, 1.9 * len(rows)),
                             squeeze=False)
    for c in range(n_classes):
        idx = np.flatnonzero(trues == c)[:max_per_class]
        panels = [_squeeze(images[idx].astype(np.float32).mean(axis=0)),
                  _prep_saliency(saliency[idx].mean(axis=0))]
        if saliency_xi is not None:
            panels.append(_prep_saliency(saliency_xi[idx].mean(axis=0)))
        for r, panel in enumerate(panels):
            axes[r, c].imshow(panel, cmap='gray' if r == 0 else 'inferno')
            axes[r, c].axis('off')
        axes[0, c].set_title(class_names[c], fontsize=8)
    for r, label in enumerate(rows):
        axes[r, 0].text(-0.12, 0.5, label, transform=axes[r, 0].transAxes,
                        rotation=90, ha='center', va='center', fontsize=8)
    fig.suptitle('Per-class means: image, |gradient| saliency'
                 + (', |gradient x input|' if saliency_xi is not None else ''))
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
