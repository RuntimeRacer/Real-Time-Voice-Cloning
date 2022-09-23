import torch
import numpy as np


def pad_tensor(self, x, pad, side='both'):
    # NB - this is just a quick method i need right now
    # i.e., it won't generalise to other shapes/dims
    b, t, c = x.size()
    total = t + 2 * pad if side == 'both' else t + pad
    if torch.cuda.is_available():
        padded = torch.zeros(b, total, c).cuda()
    else:
        padded = torch.zeros(b, total, c).cpu()
    if side == 'before' or side == 'both':
        padded[:, pad:pad + t, :] = x
    elif side == 'after':
        padded[:, :t, :] = x
    return padded


def fold_with_overlap(self, x, target, overlap):

    ''' Fold the tensor with overlap for quick batched inference.
        Overlap will be used for crossfading in xfade_and_unfold()

    Args:
        x (tensor)    : Upsampled conditioning features.
                        shape=(1, timesteps, features)
        target (int)  : Target timesteps for each index of batch
        overlap (int) : Timesteps for both xfade and rnn warmup

    Return:
        (tensor) : shape=(num_folds, target + 2 * overlap, features)

    Details:
        x = [[h1, h2, ... hn]]

        Where each h is a vector of conditioning features

        Eg: target=2, overlap=1 with x.size(1)=10

        folded = [[h1, h2, h3, h4],
                  [h4, h5, h6, h7],
                  [h7, h8, h9, h10]]
    '''

    _, total_len, features = x.size()

    # Calculate variables needed
    num_folds = (total_len - overlap) // (target + overlap)
    extended_len = num_folds * (overlap + target) + overlap
    remaining = total_len - extended_len

    # Pad if some time steps poking out
    if remaining != 0:
        num_folds += 1
        padding = target + 2 * overlap - remaining
        x = pad_tensor(x, padding, side='after')

    if torch.cuda.is_available():
        folded = torch.zeros(num_folds, target + 2 * overlap, features).cuda()
    else:
        folded = torch.zeros(num_folds, target + 2 * overlap, features).cpu()

    # Get the values for the folded tensor
    for i in range(num_folds):
        start = i * (target + overlap)
        end = start + target + 2 * overlap
        folded[i] = x[:, start:end, :]

    return folded


def xfade_and_unfold(self, y, target, overlap):

    ''' Applies a crossfade and unfolds into a 1d array.

    Args:
        y (ndarry)    : Batched sequences of audio samples
                        shape=(num_folds, target + 2 * overlap)
                        dtype=np.float64
        overlap (int) : Timesteps for both xfade and rnn warmup

    Return:
        (ndarry) : audio samples in a 1d array
                   shape=(total_len)
                   dtype=np.float64

    Details:
        y = [[seq1],
             [seq2],
             [seq3]]

        Apply a gain envelope at both ends of the sequences

        y = [[seq1_in, seq1_target, seq1_out],
             [seq2_in, seq2_target, seq2_out],
             [seq3_in, seq3_target, seq3_out]]

        Stagger and add up the groups of samples:

        [seq1_in, seq1_target, (seq1_out + seq2_in), seq2_target, ...]

    '''

    num_folds, length = y.shape
    target = length - 2 * overlap
    total_len = num_folds * (target + overlap) + overlap

    # Need some silence for the rnn warmup
    silence_len = overlap // 2
    fade_len = overlap - silence_len
    silence = np.zeros((silence_len), dtype=np.float64)

    # Equal power crossfade
    t = np.linspace(-1, 1, fade_len, dtype=np.float64)
    fade_in = np.sqrt(0.5 * (1 + t))
    fade_out = np.sqrt(0.5 * (1 - t))

    # Concat the silence to the fades
    fade_in = np.concatenate([silence, fade_in])
    fade_out = np.concatenate([fade_out, silence])

    # Apply the gain to the overlap samples
    y[:, :overlap] *= fade_in
    y[:, -overlap:] *= fade_out

    unfolded = np.zeros((total_len), dtype=np.float64)

    # Loop to add up all the samples
    for i in range(num_folds):
        start = i * (target + overlap)
        end = start + target + 2 * overlap
        unfolded[start:end] += y[i]

    return unfolded