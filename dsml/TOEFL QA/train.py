import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.autograd import Variable

import os
import numpy as np
from IPython import embed as debug

import config, utils
from data import load_dataset, WordVec
from model import AttnReader


# pylint: disable=E1101
# pylint: disable=E1120

data = load_dataset()

# Model
model = AttnReader(WordVec.instance().dim, **{
    'n_hops': 5,
    'qa_encoder_dropout': 0.8,
    'story_encoder_dropout': 0.8,
    'attn_dropout': 0.0
})
adam = optim.Adam(model.parameters(), lr=0.0003)
if config.cuda: model.cuda()
print(model)

# Training parameters
batch_size = 64
n_epochs = 100

# Save loss and accuracy for plotting
losses = []
train_accs = []
test_accs = []

#====================================================================

def plot():
    import matplotlib.pyplot as plt
    os.environ['DISPLAY'] = ':0.0' # set to local display
    plt.style.use('ggplot')
    plt.plot(range(len(losses)), losses)
    plt.savefig('figures/loss.png')
    plt.clf()
    plt.plot(range(len(train_accs)), train_accs)
    plt.plot(range(len(test_accs)), test_accs)
    plt.savefig('figures/accuracy.png')


def save():
    torch.save({
        'model': model.state_dict(),
        'adam': adam.state_dict()
    }, 'models/test.model')


def train():
    model.train()

    batches = list(data['train'].batches(batch_size))
    batches = utils.progress('Epoch {}'.format(epoch), batches)

    epoch_loss = []
    epoch_acc = []

    for step, batch in enumerate(batches):
        story, query, options, answer = utils.batch_variables(batch)

        scores = model(story, query, options, output_probs=False)
        loss = nn.CrossEntropyLoss()(scores, answer)

        model.zero_grad()
        loss.backward()
        adam.step()

        _, pred = scores.max(1)
        acc = (pred == answer).float().mean().data[0]
        loss = loss.data[0] / batch_size
        losses.append(loss)
        epoch_loss.append(loss)
        epoch_acc.append(acc)
    
    epoch_loss = np.mean(epoch_loss)
    epoch_acc = np.mean(epoch_acc)

    print('* Loss: {:.5f}'.format(epoch_loss))
    print('* Accuracy: {:.3f}'.format(epoch_acc))

    train_accs.append(epoch_acc)
    losses.append(epoch_loss)


def test():
    model.eval()

    epoch_loss = []
    epoch_acc = []

    batches = list(data['dev'].batches(batch_size))
    batches = utils.progress('Testing', batches)

    for batch in batches:
        story, query, options, answer = utils.batch_variables(batch)

        scores = model(story, query, options, output_probs=False)

        pred = scores.max(1)[1]
        acc = (pred == answer).float().mean().data[0]
        loss = nn.CrossEntropyLoss()(scores, answer)
        loss = loss.data[0] / batch_size
        epoch_loss.append(loss)
        epoch_acc.append(acc)
    
    epoch_loss = np.mean(epoch_loss)
    epoch_acc = np.mean(epoch_acc)

    print('* Loss: {:.5f}'.format(epoch_loss))
    print('* Accuracy: {:.3f}'.format(epoch_acc))

    test_accs.append(epoch_acc)

#====================================================================

try:
    for epoch in range(n_epochs):
        train()
        test()

except KeyboardInterrupt:
    pass

plot()
save()

