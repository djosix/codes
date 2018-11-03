import matplotlib.pyplot as plt
import numpy as np
import json, os

# pylint: disable=E0602

for fname in os.listdir('result'):
    path = os.path.join('result', fname)
    print(path)
    with open(path) as f:
        data = json.load(f)
    for key, value in data.items():
        exec('{} = np.array(value)'.format(key))

    plt.subplot(2, 1, 1)

    plt.title('loss and accuracy')
    plt.plot(loss, label='training')
    plt.plot(val_loss, label='validation')
    plt.ylabel('loss')
    plt.ylim(0, 2)
    plt.xticks(range(5))
    plt.legend()
    
    plt.subplot(2, 1, 2)
    plt.plot(acc * 100, label='training')
    plt.plot(val_acc * 100, label='validation')
    plt.legend()
    plt.ylabel('accuracy (%)')
    plt.ylim(0, 100)
    plt.xticks(range(5))
    
    plt.xlabel('epoch')
    plt.savefig('charts/{}.png'.format(fname))
    plt.close()
