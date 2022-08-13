import torch


def np_now(tensor):
    return tensor.detach().cpu().numpy()


def clamp(x, lo=0, hi=1):
    return max(lo, min(hi, x))


class PruneMask():
    def __init__(self, layer, prune_rnn_input):
        self.mask = []
        self.p_idx = [0]
        self.total_params = 0
        self.pruned_params = 0
        self.split_size = 0
        self.init_mask(layer, prune_rnn_input)

    def init_mask(self, layer, prune_rnn_input):
        # Determine the layer type and
        # num matrix splits if rnn
        layer_type = str(layer).split('(')[0]
        splits = {'Linear': 1, 'GRU': 3, 'LSTM': 4}

        # Organise the num and indices of layer parameters
        # Dense will have one index and rnns two (if pruning input)
        if layer_type != 'Linear':
            self.p_idx = [0, 1] if prune_rnn_input else [1]

        # Get list of parameters from layers
        params = self.get_params(layer)

        # For each param matrix in this layer, create a mask
        for W in params:
            self.mask += [torch.ones_like(W)]
            self.total_params += W.size(0) * W.size(1)

        # Need a split size for mask_from_matrix() later on
        self.split_size = self.mask[0].size(0) // splits[layer_type]

    def get_params(self, layer):
        params = []
        for idx in self.p_idx:
            params += [list(layer.parameters())[idx].data]
        return params

    def update_mask(self, layer, z, sparse_group):
        params = self.get_params(layer)
        for i, W in enumerate(params):
            self.mask[i] = self.mask_from_matrix(W, z, sparse_group)
        self.update_prune_count()

    def apply_mask(self, layer):
        params = self.get_params(layer)
        for M, W in zip(self.mask, params):
            W *= M

    def mask_from_matrix(self, W, z, sparse_group):
        # Split into gate matrices (or not)
        if self.split_size>1:
            W_split = torch.split(W, self.split_size)
        else:
            W_split = W

        M = []
        # Loop through splits
        for W in W_split:
            # Sort the magnitudes
            N = W.shape[1]

            W_abs = torch.abs(W)
            L = W_abs.reshape(W.shape[0], N // sparse_group, sparse_group)
            S = L.sum(dim=2)
            sorted_abs, _ = torch.sort(S.view(-1))

            # Pick k (num weights to zero)
            k = int(W.shape[0] * W.shape[1] // sparse_group * z)
            threshold = sorted_abs[k]
            mask = (S >= threshold).float()
            mask = mask.unsqueeze(2).expand(-1,-1,sparse_group)
            mask = mask.reshape(W.shape[0], W.shape[1])

            # Create the mask
            M += [mask]

        return torch.cat(M)

    def update_prune_count(self):
        self.pruned_params = 0
        for M in self.mask:
            self.pruned_params += int(np_now((M - 1).sum() * -1))


class Pruner(object):
    def __init__(self, start_prune, prune_steps, target_sparsity, sparse_group, prune_rnn_input=True):
        self.z = 0  # Objects sparsity @ time t
        self.t_0 = start_prune
        self.S = prune_steps
        self.Z = target_sparsity
        self.sparse_group = sparse_group
        self.num_pruned = 0
        self.total_params = 0
        self.layers = []
        self.masks = []
        self.prune_rnn_input = prune_rnn_input
        self.count_total_params()

    def update_sparsity(self, t, Z):
        z = Z * (1 - (1 - (t - self.t_0) / self.S) ** 3)
        z = clamp(z, 0, Z)
        return z

    def update_layers(self, layers, update_masks=False):
        # When using multithreaded training, model gets wrapped and split across GPUs.
        # This invalidates old references to the layers, thus we need to update them for each pruning step
        layers_new = []
        masks_new = []
        for layer in layers:
            layers_new.append((layer, self.Z))
            if update_masks:
                masks_new += [PruneMask(layer, self.prune_rnn_input)]

        self.layers = layers_new
        if update_masks:
            self.masks = masks_new

    def prune(self, step):
        for ((l,z), m) in zip(self.layers, self.masks):
            z_curr = self.update_sparsity(step, z)
            m.update_mask(l, z_curr, self.sparse_group)
            m.apply_mask(l)
        return self.count_num_pruned(), z_curr

    def restart(self, layers, step):
        # In case training is stopped
        self.update_sparsity(step)
        for ((l, z), m) in zip(layers, self.masks):
            z_curr = self.update_sparsity(step, z)
            m.update_mask(l, z_curr)

    def count_num_pruned(self):
        self.num_pruned = 0
        for m in self.masks:
            self.num_pruned += m.pruned_params
        return self.num_pruned

    def count_total_params(self):
        for m in self.masks:
            self.total_params += m.total_params
        return self.total_params