import torch
import numpy as np
from collections import defaultdict

from ptls.data_load.iterable_processing_dataset import IterableProcessingDataset

from functools import reduce
from operator import add

from ptls.data_load.feature_dict import FeatureDict
from ptls.frames.coles.split_strategy import AbsSplit
from ptls.frames.coles.metric import metric_recall_top_K, outer_cosine_similarity, outer_pairwise_distance

from ptls.frames.abs_module import ABSModule
from ptls.frames.coles.losses import ContrastiveLoss
from ptls.frames.coles.metric import BatchRecallTopK
from ptls.frames.coles.sampling_strategies import HardNegativePairSelector
from ptls.nn.head import Head
from ptls.nn.seq_encoder.containers import SeqEncoderContainer
from ptls.data_load.utils import collate_feature_dict
from ptls.data_load.padded_batch import PaddedBatch


def first(iterable, default=None):
    iterator = iter(iterable)
    return next(iterator, default)

def split_dict_of_lists_by_one(dict_of_lists):
    list_len = len(first(dict_of_lists.values()))
    to_return = [{} for _ in range(list_len)]
    
    for k, elems_list in dict_of_lists.items():
        for ind, elem in enumerate(elems_list):
            to_return[ind][k] = [elem]
            
    return to_return
    
class AmtProc(IterableProcessingDataset):
    def __init__(self, amt_cols):
        super().__init__()
        if amt_cols is None:
            amt_cols = [] 
        self._amt_cols = amt_cols
        
    def __iter__(self):
        for rec in self._src:
            features = rec[0] if type(rec) is tuple else rec
            
            for col in self._amt_cols:
                features[col] = np.nan_to_num(np.array(features[col]).astype('float32').view('float32'))
            yield rec

def metric_real_recall_top_K(X, y, K, num_pos=1, metric='cosine'):
    """
        calculate metric R@K
        X - tensor with size n x d, where n - number of examples, d - size of embedding vectors
        y - true labels
        N - count of closest examples, which we consider for recall calcualtion
        metric: 'cosine' / 'euclidean'.
            !!! 'euclidean' - to slow for datasets bigger than 100K rows
    """
    # TODO: take K from `y`
    K_adjusted = min(X.size(0) - 1, K)
    
    res = []

    n = X.size(0)
    d = X.size(1)
    max_size = 2 ** 32
    batch_size = max(1, max_size // (n * d))

    with torch.no_grad():

        for i in range(1 + (len(X) - 1) // batch_size):

            id_left = i * batch_size
            id_right = min((i + 1) * batch_size, len(y))
            y_batch = y[id_left:id_right]

            if metric == 'cosine':
                pdist = -1 * outer_cosine_similarity(X, X[id_left:id_right])
            elif metric == 'euclidean':
                pdist = outer_pairwise_distance(X, X[id_left:id_right])
            else:
                raise AttributeError(f'wrong metric "{metric}"')

            values, indices = pdist.topk(K_adjusted + 1, 0, largest=False)

            y_rep = y_batch.repeat(K_adjusted, 1)
            res.append((y[indices[1:]] == y_rep).sum().item())

    return np.sum(res) / len(y) / num_pos

def cosine_similarity_matrix(x1, x2):
    x1_norm = x1 / x1.norm(dim=1)[:, None]
    x2_norm = x2 / x2.norm(dim=1)[:, None]
    return torch.mm(x1_norm, x2_norm.transpose(0, 1))

def metric_recall_top_K_for_embs(embs_1, embs_2, true_matches, K=100):
    similarity_matrix = cosine_similarity_matrix(embs_1, embs_2)
    K_adjusted = min(len(embs_1), K)
    top_k = similarity_matrix.topk(k=K_adjusted, dim=1).indices
    correct_matches = 0
    for i, indices in enumerate(top_k):
        if true_matches[i] in indices:
            correct_matches += 1
    recall_at_k = correct_matches / len(similarity_matrix)
    return recall_at_k

class M3CoLESModule(ABSModule):
    """
    Multi-Modal Matching
    Contrastive Learning for Event Sequences ([CoLES](https://arxiv.org/abs/2002.08232))

    Subsequences are sampled from original sequence.
    Samples from the same sequence are `positive` examples
    Samples from the different sequences are `negative` examples
    Embeddings for all samples are calculated.
    Paired distances between all embeddings are calculated.
    The loss function tends to make positive distances smaller and negative ones larger.

    Parameters
        seq_encoder:
            Model which calculate embeddings for original raw transaction sequences
            `seq_encoder` is trained by `CoLESModule` to get better representations of input sequences
        head:
            Model which helps to train. Not used during inference
            Can be normalisation layer which make embedding l2 length equals 1
            Can be MLP as `projection head` like in SymCLR framework.
        loss:
            loss object from `ptls.frames.coles.losses`.
            There are paired and triplet loss. They are required sampling strategy
            from `ptls.frames.coles.sampling_strategies`. Sampling strategy takes a relevant pairs or triplets from
            pairwise distance matrix.
        validation_metric:
            Keep None. `ptls.frames.coles.metric.BatchRecallTopK` used by default.
        optimizer_partial:
            optimizer init partial. Network parameters are missed.
        lr_scheduler_partial:
            scheduler init partial. Optimizer are missed.

    """
    def __init__(self,
                 seq_encoders=None,
                 mod_names=None,
                 head=None,
                 loss=None,
                 validation_metric=None,
                 optimizer_partial=None,
                 lr_scheduler_partial=None):
        torch.set_float32_matmul_precision('high')
        if head is None:
            head = Head(use_norm_encoder=True)

        if loss is None:
            loss = ContrastiveLoss(margin=0.5,
                                   sampling_strategy=HardNegativePairSelector(neg_count=5))

        if validation_metric is None:
            validation_metric = BatchRecallTopK(K=4, metric='cosine')
        
        for k in seq_encoders.keys():
            if type(seq_encoders[k]) is str:
                seq_encoders[k] = seq_encoders[seq_encoders[k]]
                
        super().__init__(validation_metric,
                         first(seq_encoders.values()),
                         loss,
                         optimizer_partial,
                         lr_scheduler_partial)
        
        self.seq_encoders = torch.nn.ModuleDict(seq_encoders)
        self._head = head   
        self.y_h_cache = {'train':[], 'valid': []}
        
    @property
    def metric_name(self):
        return 'recall_top_k'

    @property
    def is_requires_reduced_sequence(self):
        return True
    
    def forward(self, x):
        res = {}
        
        for mod_name in x.keys():
            res[mod_name] = self.seq_encoders[mod_name](x[mod_name])
            
        return res

    def shared_step(self, x, y):
        y_h = self(x)
        
        if self._head is not None:
            y_h_head = {k: self._head(y_h_k) for k, y_h_k in y_h.items()}
            y_h = y_h_head
            
        return y_h, y
    
    def _one_step(self, batch, _, stage):
        y_h, y = self.shared_step(*batch)
        y_h_list = list(y_h.values())
        loss = self._loss(torch.cat(y_h_list), torch.cat([y, y]))
        self.log(f'loss/{stage}', loss.detach())
        
        x, y = batch
        for mod_name, mod_x in x.items():
            self.log(f'seq_len/{stage}/{mod_name}', x[mod_name].seq_lens.float().mean().detach(), prog_bar=True)
        
        if stage == "valid":
            n, d = y_h_list[0].shape
            y_h_concat = torch.zeros((2*n, d), device = y_h_list[0].device)
            
            for i in range(2):
                y_h_concat[range(i,2*n,2)] = y_h_list[i] 
            
            if len(self.y_h_cache[stage]) <= 380:
                self.y_h_cache[stage].append((y_h_concat.cpu(), {k: y_h_k.cpu() for k, y_h_k in y_h.items()} , 
                                             {k:x_k.seq_lens.cpu() for k, x_k in x.items()})) 
    
        return loss
    
    def training_step(self, batch, _):
        return self._one_step(batch, _, "train")
    
    def validation_step(self, batch, _):
        return self._one_step(batch, _, "valid")
    
    def on_validation_epoch_end(self):        
        #len_intervals = [(0, 10), (10, 20), (20, 30), (30, 40), (40, 60), (60, 80), (80, 120), (120, 160), (160, 240)]
        self.log_recall_top_K(self.y_h_cache['valid'], len_intervals=None, stage="valid", K=100)
        self.log_recall_top_K(self.y_h_cache['valid'], len_intervals=None, stage="valid", K=50)
        self.log_recall_top_K(self.y_h_cache['valid'], len_intervals=None, stage="valid", K=1)
        
        
        del self.y_h_cache["valid"]
        self.y_h_cache["valid"] = []
        
    def log_recall_top_K(self, y_h_cache, len_intervals=None, stage="valid", K=100):
        y_h = torch.cat([item[0] for item in y_h_cache], dim = 0)
        y_h_mods = defaultdict(list)
        seq_lens_dict = defaultdict(list)
        
        for item in y_h_cache:
            for k, emb in item[1].items():
                y_h_mods[k].append(emb)
                
            for k, l in item[2].items():
                seq_lens_dict[k].append(l)
        
        y_h_mods = {k: torch.cat(el, dim=0) for k ,el in y_h_mods.items()}
        seq_lens_dict = {k: torch.cat(el) for k ,el in seq_lens_dict.items()}

        #n, _ = y_h.shape
        #y = torch.zeros((n,)).cpu().long()
        #y[range(0,n,2)] = torch.arange(0, n//2)
        #y[range(1,n,2)] = torch.arange(0, n//2)
        #computed_metric = metric_real_recall_top_K(y_h, y, K=100)
        y_h_bank, y_h_rmb = list(y_h_mods.values())
        computed_metric_b2r = metric_recall_top_K_for_embs(y_h_bank, y_h_rmb, torch.arange(y_h_rmb.shape[0]), K=K)
        computed_metric_r2b = metric_recall_top_K_for_embs(y_h_rmb, y_h_bank, torch.arange(y_h_rmb.shape[0]), K=K)
        
        if len_intervals != None:
            for mod, seq_lens in seq_lens_dict.items():
                for start, end in len_intervals:
                    mask = ((seq_lens > start) & (seq_lens <= end))

                    if torch.any(mask):
                        #y_h_filtered = y_h[mask.repeat_interleave(2)]
                        y_h_bank_filtered = y_h_bank[mask]
                        y_h_rmb_filtered = y_h_rmb[mask]

                        #y = torch.div(torch.arange(len(y_h_filtered)), 2, rounding_mode='floor')
                        #recall = metric_real_recall_top_K(y_h_filtered, y, K=100)
                        recall_r2b = metric_recall_top_K_for_embs(y_h_rmb_filtered, y_h_bank_filtered, torch.arange(y_h_rmb_filtered.shape[0]), K=100)
                        recall_b2r = metric_recall_top_K_for_embs(y_h_bank_filtered, y_h_rmb_filtered, torch.arange(y_h_rmb_filtered.shape[0]), K=100)

                        #self.log(f"{mode}/R@100_len_from_{start}_to_{end}", recall, prog_bar=True)
                        self.log(f"{stage}/{mod}/r2b_R@100_len_from_{start}_to_{end}", recall_r2b, prog_bar=True)
                        self.log(f"{stage}/{mod}/b2r_R@100_len_from_{start}_to_{end}", recall_b2r, prog_bar=True)
        
        #self.log(f"{mode}/R@100", computed_metric, prog_bar=True)
        self.log(f"{stage}/click2trx_R@{K}", computed_metric_r2b, prog_bar=True)
        self.log(f"{stage}/trx2click_R@{K}", computed_metric_b2r, prog_bar=True)

# ----------------------------
def collate_feature_dict(batch):
    new_x_ = defaultdict(list)
    for i, x in enumerate(batch):
        for k, v in x.items():
            new_x_[k].append(v)
    
    seq_col = next(k for k, v in batch[0].items() if FeatureDict.is_seq_feature(k, v))
    lengths = torch.LongTensor([len(rec[seq_col]) for rec in batch])
    new_x = {}
    for k, v in new_x_.items():
        if type(v[0]) is torch.Tensor:
            if k.startswith('target'):
                new_x[k] = torch.stack(v, dim=0)
            else:
                new_x[k] = torch.nn.utils.rnn.pad_sequence(v, batch_first=True)
        elif type(v[0]) is np.ndarray:
            new_x[k] = v  # list of arrays[object]
        else:
            v = np.array(v)
            if v.dtype.kind == 'i':
                new_x[k] = torch.from_numpy(v).long()
            elif v.dtype.kind == 'f':
                new_x[k] = torch.from_numpy(v).float()
            elif v.dtype.kind == 'b':
                new_x[k] = torch.from_numpy(v).bool()
            else:
                new_x[k] = v
    return PaddedBatch(new_x, lengths)

    
def collate_multimodal_feature_dict(batch):
    res = {}
    for source, source_batch in batch.items():
        res[source] = collate_feature_dict(source_batch)
    return res
    
def get_dict_class_labels(batch):
    res = defaultdict(list)
    for i, samples in enumerate(batch):
        for source, values in samples.items():
            for _ in values:
                res[source].append(i)
    for source in res:
        res[source] = torch.LongTensor(res[source])
    return dict(res)
            

class MultiModalDiffSplitDataset(FeatureDict, torch.utils.data.Dataset):
    def __init__(
        self,
        data,
        splitters,
        source_features,
        col_id,
        source_names,
        col_time='event_time',
        *args, **kwargs
    ):
        """
        Dataset for multimodal learning.
        Parameters:
        -----------
        data:
            concatinated data with feature dicts.
        splitter:
            object from from `ptls.frames.coles.split_strategy`.
            Used to split original sequence into subsequences which are samples from one client.
        source_features:
            list of column names 
        col_id:
            column name with user_id
        source_names:
            column name with name sources
        col_time:
            column name with event_time
        """
        super().__init__(*args, **kwargs)
        
        self.data = data
        self.splitters = splitters
        self.col_time = col_time
        self.col_id = col_id
        self.source_names = source_names
        self.source_features = source_features
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        feature_arrays = self.data[idx]
        split_data = self.split_source(feature_arrays)
        return self.get_splits(split_data)
    
    def __iter__(self):
        for feature_arrays in self.data:
            split_data = self.split_source(feature_arrays)
            yield self.get_splits(split_data)
            
    def split_source(self, feature_arrays):
        res = defaultdict(dict)
        for feature_name, feature_array in feature_arrays.items():
            if feature_name == self.col_id:
                res[self.col_id] = feature_array
            else:
                source_name, feature_name_transform = self.get_names(feature_name)
                res[source_name][feature_name_transform] = feature_array
        for source in self.source_names:
            if source not in res:
                res[source] = {source_feature: torch.tensor([]) for source_feature in self.source_features[source]}
        return res
    
    def get_names(self, feature_name):
        idx_del = feature_name.find('_')
        return feature_name[:idx_del], feature_name[idx_del + 1:]
    
    def get_splits(self, feature_arrays):
        res = {}
        for source_name, feature_array in feature_arrays.items():
            if source_name != self.col_id:
                local_date = feature_array[self.col_time]
                if source_name not in self.splitters:
                    continue
                indexes = self.splitters[source_name].split(local_date)
                res[source_name] = [{k: v[ix] for k, v in feature_array.items() if self.is_seq_feature(k, v)} for ix in indexes]
        return res
        
    def collate_fn(self, batch, return_dct_labels=False):
        dict_class_labels = get_dict_class_labels(batch)
        batch = reduce(lambda x, y: {k: x[k] + y[k] for k in x if k in y}, batch)
        padded_batch = collate_multimodal_feature_dict(batch)
        if return_dct_labels:
            return padded_batch, dict_class_labels
        return padded_batch, dict_class_labels[list(dict_class_labels.keys())[0]]

    
class MultiModalDiffSplitIterableDataset(MultiModalDiffSplitDataset, torch.utils.data.IterableDataset):
    pass