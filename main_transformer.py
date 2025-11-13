# https://towardsdatascience.com/a-detailed-guide-to-pytorchs-nn-transformer-module-c80afbc9ffb1
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, RandomSampler

from language_processor import MAX_LENGTH, EOS_token, SOS_token, prepareData
from transformer_model import Transformer
from utils import showPlot, timeSince

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ================================================== #
# Preparing Training Data
def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]


def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(1, -1)


def get_dataloader(batch_size, target_lang='fra'):
    input_lang, output_lang, pairs = prepareData('eng', target_lang, True)

    n = len(pairs)
    input_ids = np.zeros((n, MAX_LENGTH), dtype=np.int32)
    target_ids = np.zeros((n, MAX_LENGTH + 1), dtype=np.int32)  # Because it is transformer, extra space to shift

    for idx, (inp, tgt) in enumerate(pairs):
        inp_ids = indexesFromSentence(input_lang, inp)
        tgt_ids = indexesFromSentence(output_lang, tgt)
        inp_ids.append(EOS_token)
        tgt_ids.append(EOS_token)
        input_ids[idx, :len(inp_ids)] = inp_ids
        target_ids[idx, :len(tgt_ids)] = tgt_ids

    train_data = TensorDataset(torch.LongTensor(input_ids).to(device),
                               torch.LongTensor(target_ids).to(device))

    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
    return input_lang, output_lang, train_dataloader, pairs


# ================================================== #
# Training the Model
def train_epoch(dataloader, transformer, opt, loss_fn):
    total_loss = 0

    for batch in dataloader:
        X, y = batch

        # Now we shift the tgt by one so with the <SOS> we predict the token at pos 1
        y_sos = torch.zeros((y.shape[0], 1), dtype=y.dtype).fill_(SOS_token).to(device)
        y_input = torch.cat((y_sos, y[:, :-1]), dim=1)
        y_expected = y

        x_valid_mask = transformer.create_pad_mask(X, 0)
        y_valid_mask = torch.cat(
            (transformer.create_pad_mask(y_input[:, :1], 1), transformer.create_pad_mask(y_input[:, 1:], 0)), dim=1)

        # Get mask to mask out the next words
        sequence_length = y_input.size(1)
        tgt_mask = transformer.get_tgt_mask(sequence_length).to(device)

        # Standard training except we pass in y_input and tgt_mask
        pred = transformer(X, y_input, tgt_mask, src_pad_mask=x_valid_mask, tgt_pad_mask=y_valid_mask)

        # Permute pred to have batch size first again
        pred = pred.permute(1, 2, 0)
        loss = loss_fn(pred, y_expected)

        opt.zero_grad()
        loss.backward()
        opt.step()

        total_loss += loss.detach().item()

    return total_loss / len(dataloader)


# ================================================== #
def train(train_dataloader, transformer, n_epochs, learning_rate=0.001, print_every=100, plot_every=100):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    optimizer = torch.optim.Adam(transformer.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    transformer.train()

    for epoch in range(1, n_epochs + 1):
        loss = train_epoch(train_dataloader, transformer, optimizer, criterion)
        print_loss_total += loss
        plot_loss_total += loss

        if epoch % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, epoch / n_epochs),
                                         epoch, epoch / n_epochs * 100, print_loss_avg))

        if epoch % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

    showPlot(plot_losses)
    return


# ================================================== #
# Evaluation
def evaluate(transformer, sentence, input_lang, output_lang):
    with torch.no_grad():
        input_tensor = tensorFromSentence(input_lang, sentence[0]).to(device)
        target_tensor = torch.tensor([SOS_token], dtype=torch.long, device=device).view(1, -1).to(device)

        X = torch.zeros((1, MAX_LENGTH), dtype=input_tensor.dtype).to(device)
        X[0, :len(input_tensor[0])] = input_tensor[0]

        x_valid_mask = transformer.create_pad_mask(X, 0)

        decoded_words = ['']
        i = 0
        while not decoded_words[-1] == 'EOS' and i < MAX_LENGTH:
            tgt_mask = transformer.get_tgt_mask(target_tensor.size(1)).to(device)
            pred = transformer(X, target_tensor, tgt_mask, src_pad_mask=x_valid_mask)

            # Permute pred to have batch size first again
            pred = pred.permute(1, 2, 0)

            output_topk = pred.topk(1, dim=1)
            decoded_words.append(output_lang.index2word[output_topk[1][0][0][-1].item()])
            target_next = output_topk[1][:, 0, -1]
            if target_next.ndim == 1:
                target_next = target_next.unsqueeze(0)
            target_tensor = torch.cat((target_tensor, target_next), dim=1)
            i += 1

    return decoded_words[1:]


def evaluateRandomly(transformer, n=10):
    for i in range(n):
        pair = random.choice(pairs)
        print('>', pair[0])
        print('=', pair[1])
        output_words = evaluate(transformer, pair, input_lang, output_lang)
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')
    return


if __name__ == '__main__':
    batch_size = 16
    n_epochs = 600
    target_lang = 'kor'
    # target_lang = 'fra'

    input_lang, output_lang, train_dataloader, pairs = get_dataloader(batch_size, target_lang=target_lang)

    transformer = Transformer(
        num_tokens_src=input_lang.n_words, num_tokens_tgt=output_lang.n_words, dim_model=32, num_heads=4,
        num_encoder_layers=1, num_decoder_layers=1, dropout_p=0.1
    ).to(device)

    train(train_dataloader, transformer, n_epochs)

    transformer.eval()
    evaluateRandomly(transformer)
