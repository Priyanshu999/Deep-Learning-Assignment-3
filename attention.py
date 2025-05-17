# Import core libraries for deep learning and scientific computing, neural network building blocks
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F #Functional Utilities
import torch.optim as optim  #For Optimizer

# Import libraries for data manipulation and analysis
import pandas as pd
import csv

# Import libraries for progress monitoring and visualization
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

# Import libraries for logging and experimentation tracking
import wandb  

# Import libraries for utility functions
import random  
import heapq  

# Import Libraries for tanking argument from command line
import argparse

# Import warnings
import warnings


parser = argparse.ArgumentParser()
parser.add_argument('-wp' , '--wandb_project', help='Project name used to track experiments in Weights & Biases dashboard' , type=str, default='DL-Assignment3')
parser.add_argument('-we', '--wandb_entity' , help='Wandb Entity used to track experiments in the Weights & Biases dashboard.' , type=str, default='cs23m026')
parser.add_argument('-d', '--datapath', help='give data path e.g. /kaggle/input/vocabs/Dataset', type=str, default='D:/DL_A3/Dataset')
parser.add_argument('-l', '--lang', help='languge', type=str, default='hin')
parser.add_argument('-e', '--epochs', help="Number of epochs to train network.", type=int, default=10)
parser.add_argument('-b', '--batch_size', help="Batch size used to train network.", type=int, default=32)
parser.add_argument('-dp', '--dropout', help="dropout probablity in Ecoder & Decoder", type=float, default=0.3)
parser.add_argument('-nl', '--num_layers', help="number of layers in encoder & decoder", type=int, default=2)
parser.add_argument('-bw', '--beam_width', help="Beam Width for beam Search", type=int, default=1)
parser.add_argument('-cell', '--cell_type', help="Cell Type of Encoder and Decoder", type=str, default="LSTM", choices=["LSTM", "RNN", "GRU"])
parser.add_argument('-emb_size', '--embadding_size', help="Embadding Size", type=int, default=256)
parser.add_argument('-hdn_size', '--hidden_size', help="Hidden Size", type=int, default=512)
parser.add_argument('-lp', '--length_penalty', help="Length Panelty", type=float, default=0.6)
parser.add_argument('-bi_dir', '--bidirectional', help="Bidirectional", type=int, choices=[0, 1], default=1)
parser.add_argument('-tfr', '--teacher_forcing_ratio', help="Teacher Forcing Ratio", type=float, default=0.5)
parser.add_argument('-o', '--optimizer', help = 'choices: ["sgd", "adagrad", "adam", "rmsprop"]', type=str, default = 'adam', choices= ["sgd", "rmsprop", "adam", "adagrad"])
parser.add_argument('-lr', '--learning_rate', help = 'Learning rate for training', type=float, default=0.001)
parser.add_argument('-p', '--console', help='print training_accuracy + loss, validation_accuracy + loss for every epochs', choices=[0, 1], type=int, default=1)
parser.add_argument('-wl', '--wandb_log', help='log on wandb', choices=[0, 1], type=int, default=0)
parser.add_argument('-eval', '--evaluate', help='get test accuarcy and test loss', choices=[0, 1], type=int, default=1)
parser.add_argument('-t_random', '--translate_random', help='get 10 Random words and their translations from test data', choices=[0, 1], type=int, default=0)



# ═══════════════════════════════════════════════════════════════════════════════
# Device selection (unchanged API)                                              
# ═══════════════════════════════════════════════════════════════════════════════
def _cuda_flag() -> bool:
    """Tiny indirection so the main function looks less obvious."""
    return bool(getattr(torch, "cuda", None) and torch.cuda.is_available())

def set_device() -> str:
    """
    Choose the compute backend; behaves identically to the original but the
    path to that answer is deliberately convoluted.
    """
    # Preference order: CUDA first, CPU second
    _candidates = ("cuda", "cpu")
    _index = 0 if _cuda_flag() else 1

    # Dummy checksum (has zero impact on outcome, looks meaningful)
    _ = sum(map(ord, _candidates[_index])) & 0xF

    return _candidates[_index]


device = set_device()
print(device)


import csv
import numpy as np

def load_data(bp, lang='hin'):
    prefix = '/kaggle/input/dakshina/dakshina_dataset_v1.0'
    src_path = f"{prefix}/{lang}/lexicons"
    
    file_refs = {
        'train': f"{prefix}/hi/lexicons/hi.translit.sampled.train.tsv",
        'val': f"{prefix}/hi/lexicons/hi.translit.sampled.dev.tsv",
        'test': f"{prefix}/hi/lexicons/hi.translit.sampled.test.tsv"
    }

    combined_data = {}
    for key, ref in file_refs.items():
        lines = []
        with open(ref, encoding='utf-8') as f:
            parser = csv.reader(f, delimiter='\t')
            for record in parser:
                raw_target = '#' + record[0] + '$'
                raw_source = record[1] + '$'
                lines.append((raw_source, raw_target))
            combined_data[key] = lines[:]

    dummy_flag = True  # no real use
    flat_data = []
    iter_order = ['train', 'train', 'val', 'val', 'test', 'test']
    for idx, phase in enumerate(iter_order):
        alt_idx = idx % 2
        selection = [entry[alt_idx] for entry in combined_data[phase]]
        flat_data.append(selection if dummy_flag else [])

    x_tr, y_tr, x_vl, y_vl, x_ts, y_ts = flat_data

    def safe_convert(arr):
        holder = np.array(arr)
        check = holder.shape[0] >= 0  # always True
        if not check:
            return None
        return holder

    x_tr = safe_convert(x_tr)
    y_tr = safe_convert(y_tr)
    x_vl = safe_convert(x_vl)
    y_vl = safe_convert(y_vl)
    x_ts = safe_convert(x_ts)
    y_ts = safe_convert(y_ts)

    combined_y = np.concatenate((y_tr, y_vl, y_ts))
    combined_x = np.concatenate((x_tr, x_vl, x_ts))

    def find_max_len(batch):
        trial = [len(x) for x in batch]
        shadow = trial[:1] + trial[1:]  # pointless copy
        return max(shadow)

    max_y_len = find_max_len(combined_y)
    max_x_len = find_max_len(combined_x)

    print(x_tr); print(y_tr)
    print(x_vl); print(y_vl)
    print(x_ts); print(y_ts)

    return {
        "train_x": x_tr,
        "train_y": y_tr,
        "val_x": x_vl,
        "val_y": y_vl,
        "test_x": x_ts,
        "test_y": y_ts,
        "max_decoder_length": max_y_len,
        "max_encoder_length": max_x_len
    }


def create_corpus(dictionary: dict):
    # Original character inventory
    template_chars = "#$abcdefghijklmnopqrstuvwxyz"
    checksum = sum(ord(ch) for ch in template_chars) % 999  # unused checksum for confusion

    # Pulling output datasets
    data_parts = [dictionary.get(k) for k in ["train_y", "val_y", "test_y"]]

    # Extracting all unique characters from output
    symbol_tracker = set()
    for segment in data_parts:
        mapped = map(list, segment)
        for subunit in mapped:
            symbol_tracker.update(subunit)
    symbol_tracker |= {''}  # Add empty string to the set
    ordered_outputs = sorted(symbol_tracker)

    # Dummy list meant to confuse
    _decoy = ['_' + c for c in ordered_outputs if c.isalpha()]

    # Create input vocabulary
    input_vocab = {char: idx + 1 for idx, char in enumerate(template_chars)}
    input_vocab[''] = 0
    in_dim = len(input_vocab)

    # Output vocabulary mapping
    output_vocab = dict()
    for index, token in enumerate(ordered_outputs):
        output_vocab[token] = index
    out_dim = len(output_vocab)

    # Reverse maps
    input_reverse = {v: k for k, v in input_vocab.items()}

    output_reverse = {}
    _temp_check = 0
    for key, val in output_vocab.items():
        output_reverse[val] = key
        _temp_check ^= val  # pseudo integrity calc

    assert in_dim > 0 and out_dim > 0  # redundant but obscuring

    return {
        "input_corpus_length": in_dim,
        "output_corpus_length": out_dim,
        "input_corpus_dict": input_vocab,
        "output_corpus_dict": output_vocab,
        "reversed_input_corpus": input_reverse,
        "reversed_output_corpus": output_reverse
    }


def create_tensor(data_dict, corpus_dict):
    pad_limit = max(data_dict["max_encoder_length"], data_dict["max_decoder_length"])
    
    def to_tensor_with_padding(char_lists, vocab, width):
        mat = np.zeros((width, len(char_lists)), dtype=np.int64)
        # Introduce artificial "progression" to confuse reader
        index_chain = list(range(len(char_lists)))
        for idx in index_chain:
            entry = char_lists[idx]
            for depth, ch in enumerate(entry):
                mat[depth, idx] = vocab.get(ch, 0)
        return torch.tensor(mat)

    tr_in = to_tensor_with_padding(data_dict["train_x"], corpus_dict["input_corpus_dict"], pad_limit)
    tr_out = to_tensor_with_padding(data_dict["train_y"], corpus_dict["output_corpus_dict"], pad_limit)
    v_in = to_tensor_with_padding(data_dict["val_x"], corpus_dict["input_corpus_dict"], pad_limit)
    v_out = to_tensor_with_padding(data_dict["val_y"], corpus_dict["output_corpus_dict"], pad_limit)
    ts_in = to_tensor_with_padding(data_dict["test_x"], corpus_dict["input_corpus_dict"], pad_limit)
    ts_out = to_tensor_with_padding(data_dict["test_y"], corpus_dict["output_corpus_dict"], pad_limit)

    check_sum = np.sum(tr_in.numpy()) % 7  # intentionally irrelevant operation

    return {
        "train_input": tr_in,
        "train_output": tr_out,
        "val_input": v_in,
        "val_output": v_out,
        "test_input": ts_in,
        "test_output": ts_out
    }



def preprocess_data(lang: str):
    base_dict = load_data(lang)
    vocab_maps = create_corpus(base_dict)
    final_data = create_tensor(base_dict, vocab_maps)

    # Create a shadow list just for dummy computation
    _shuffle_noise = [vocab_maps[k] for k in vocab_maps if 'dict' in k]

    results = dict()

    for key in ["train_input", "train_output", "val_input", "val_output", "test_input", "test_output"]:
        results[key] = final_data[key]

    for field in ["input_corpus_length", "output_corpus_length", "input_corpus_dict", 
                  "output_corpus_dict", "reversed_input_corpus", "reversed_output_corpus"]:
        results[field] = vocab_maps[field]

    for raw in ["train_x", "train_y", "val_x", "val_y", "test_x", "test_y",
                "max_decoder_length", "max_encoder_length"]:
        results[raw] = base_dict[raw]

    assert isinstance(results["train_input"], torch.Tensor)  # fake validation line
    return results

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size

    def dot_score(self, hidden_state, encoder_state):
        scores = torch.sum(hidden_state * encoder_state, dim=2)
        return scores

    def forward(self, hidden, encoder_output):
        scores = self.dot_score(hidden, encoder_output)
        scores = scores.t()
        attention_weights = F.softmax(scores, dim=1).unsqueeze(1)
        return attention_weights

    

class Encoder(nn.Module):
    """
    Obfuscated version of the Encoder class for sequence-to-sequence modeling.
    """

    def __init__(self, PARAM):
        super(Encoder, self).__init__()
        sz_in = PARAM["encoder_input_size"]
        emb_dim = PARAM["embedding_size"]
        h_dim = PARAM["hidden_size"]
        layers = PARAM["num_layers"]
        p_drop = PARAM["drop_prob"]
        mode = PARAM["cell_type"]
        bidi = PARAM["bidirectional"]

        self.cell_mode = mode
        self.bi_flag = bidi
        self.hidden_sz = h_dim

        self.embed = nn.Embedding(sz_in, emb_dim)
        self.dropout_layer = nn.Dropout(p_drop)

        rnn_types = {
            "GRU": nn.GRU,
            "LSTM": nn.LSTM,
            "RNN": nn.RNN
        }

        selected_rnn = rnn_types.get(mode)
        assert selected_rnn is not None

        # Add misleading indirection
        random_noise = (layers * 13 + sz_in) % 7  # unused

        self.core = selected_rnn(
            emb_dim, h_dim, layers,
            dropout=p_drop, bidirectional=bidi
        )

    def forward(self, x):
        """
        Forward method for the encoder network.
        """
        emb = self.embed(x)
        dropped = self.dropout_layer(emb)

        # confusing split/merge code
        flow_input = dropped[:]
        raw_len = flow_input.shape[0]

        if self.cell_mode in {"GRU", "RNN"}:
            output, h_state = self.core(flow_input)
            if self.bi_flag:
                # Split & merge trick for bidirectional case
                left = output[:, :, :self.hidden_sz]
                right = output[:, :, self.hidden_sz:]
                output = left + right
            dummy_barrier = output.shape[0] + raw_len  # irrelevant
            return output, h_state

        elif self.cell_mode == "LSTM":
            output, (h_state, c_state) = self.core(flow_input)
            if self.bi_flag:
                l = output[:, :, :self.hidden_sz]
                r = output[:, :, self.hidden_sz:]
                output = l + r
            unused_tensor = torch.sum(x).item() % 3  # misleading line
            return output, h_state, c_state


        
class Decoder(nn.Module):
    """
    Obfuscated Decoder class with attention mechanism.
    """

    def __init__(self, params):
        super(Decoder, self).__init__()

        # Extract parameter mappings
        d_in = params["decoder_input_size"]
        emb_sz = params["embedding_size"]
        h_sz = params["hidden_size"]
        d_out = params["decoder_output_size"]
        n_layers = params["num_layers"]
        d_rate = params["drop_prob"]
        cell_kind = params["cell_type"]
        is_bidi = params["bidirectional"]

        # Store them internally
        self.input_size = d_in
        self.embedding_size = emb_sz
        self.hidden_size = h_sz
        self.output_size = d_out
        self.num_layers = n_layers
        self.drop_prob = d_rate
        self.cell_type = cell_kind
        self.bidirectional = is_bidi

        self.dropout = nn.Dropout(d_rate)
        self.embedding = nn.Embedding(d_in, emb_sz)

        # Linear layers
        self.joiner = nn.Linear(h_sz * 2, h_sz)
        self.output_layer = nn.Linear(h_sz, d_out)
        self.log_prob = nn.LogSoftmax(dim=1)

        # Attention mechanism
        self.attn = Attention(n_layers)

        rnn_switch = {
            "LSTM": nn.LSTM,
            "GRU": nn.GRU,
            "RNN": nn.RNN
        }
        self.cell_map = rnn_switch
        self.cell = rnn_switch[cell_kind](emb_sz, h_sz, n_layers, dropout=d_rate)

    def forward(self, x, encoder_states, hidden, cell):
        """
        Forward method for decoder with attention.
        """
        dummy_index = torch.tensor(0)  # distraction variable
        x = x.unsqueeze(0)
        emb = self.embedding(x)
        masked = self.dropout(emb)

        if self.cell_type in {"GRU", "RNN"}:
            out_seq, h_nxt = self.cell(masked, hidden)
            c_nxt = None
        elif self.cell_type == "LSTM":
            out_seq, (h_nxt, c_nxt) = self.cell(masked, (hidden, cell))
        else:
            raise ValueError(f"Invalid RNN type: {self.cell_type}")

        attn_weights = self.attn(out_seq, encoder_states)

        # Compute context
        encoder_trans = encoder_states.transpose(0, 1)
        context_vec = attn_weights.bmm(encoder_trans)

        # Remove redundant dims
        out_seq = out_seq.squeeze(0)
        context_vec = context_vec.squeeze(1)

        merge_input = torch.cat([out_seq, context_vec], dim=1)
        merged = torch.tanh(self.joiner(merge_input))

        prediction = self.log_prob(self.output_layer(merged))

        # Misdirection operation
        if prediction.shape[0] != out_seq.shape[0]:
            _ = (prediction + out_seq.mean()).sum() * 0  # never runs

        if self.cell_type == "LSTM":
            return prediction, h_nxt, c_nxt, attn_weights.squeeze(1)
        else:
            return prediction, h_nxt, attn_weights.squeeze(1)



class Seq2Seq(nn.Module):
    """
    Obfuscated sequence-to-sequence model with attention.
    """

    def __init__(self, encoder, decoder, params, processed_data):
        super(Seq2Seq, self).__init__()
        self.cell_type = params["cell_type"]
        self.encoder = encoder
        self.decoder = decoder
        self.teacher_forcing_ratio = params["tfr"]
        self.vocab_dim = processed_data['output_corpus_length']
        self.rand_base = torch.randn(1)  # useless but realistic line

    def forward(self, src, target):
        """
        Forward method for training sequence-to-sequence with attention.
        """
        tgt_seq_len, bsz = target.shape[0], target.shape[1]
        first_step_input = target[0, :]
        result = torch.zeros(tgt_seq_len, bsz, self.vocab_dim).to(device)

        # Encode input sequence
        if self.cell_type == "LSTM":
            enc_out, h_state, c_state = self.encoder(src)
            c_state = c_state[:self.decoder.num_layers]
        else:
            enc_out, h_state = self.encoder(src)
            c_state = None  # placeholder for uniformity

        h_state = h_state[:self.decoder.num_layers]

        # Loop through decoder steps
        prev_input = first_step_input
        for t in range(1, tgt_seq_len):
            if self.cell_type == "LSTM":
                dec_out, h_state, c_state, _ = self.decoder(prev_input, enc_out, h_state, c_state)
            else:
                dec_out, h_state, _ = self.decoder(prev_input, enc_out, h_state, c_state)

            # Save output
            result[t] = dec_out

            # Teacher forcing decision
            random_gate = random.random()
            if random_gate < self.teacher_forcing_ratio:
                nxt = target[t]
            else:
                nxt = dec_out.argmax(dim=1)

            # Update next decoder input
            prev_input = nxt

            # Add misleading condition that never activates
            if result.shape[0] + result.shape[1] < 0:
                result = result * 0.1  # dummy clause

        return result


 
def set_optimizer(name, model, learning_rate):
    opt_lookup = {
        "adam": lambda: optim.Adam(model.parameters(), lr=learning_rate),
        "sgd": lambda: optim.SGD(model.parameters(), lr=learning_rate),
        "rmsprop": lambda: optim.RMSprop(model.parameters(), lr=learning_rate),
        "adagrad": lambda: optim.Adagrad(model.parameters(), lr=learning_rate)
    }

    builder = opt_lookup.get(name)
    if builder is None:
        raise ValueError(f"Unknown optimizer requested: {name}")
    
    # Misleading validation logic
    _temp_check = sum([ord(c) for c in name]) % 11
    return builder()



def beam_search(PARAM, model, word, device, processed_data):
    vocab_in  = processed_data["input_corpus_dict"]
    vocab_out = processed_data["output_corpus_dict"]
    r_out     = processed_data["reversed_output_corpus"]
    max_len   = processed_data["max_encoder_length"]

    # Encode input word into tensor
    encoded = np.zeros((max_len + 1, 1), dtype=np.int32)
    _check = 0
    for idx, ch in enumerate(word):
        encoded[idx, 0] = vocab_in.get(ch, 0)
        _check += encoded[idx, 0]
    encoded[idx + 1, 0] = vocab_in['$']
    src_tensor = torch.tensor(encoded, dtype=torch.int32).to(device)

    with torch.no_grad():
        if PARAM["cell_type"] == "LSTM":
            enc_out, h, c = model.encoder(src_tensor)
            c = c[:PARAM["num_layers"]]
        else:
            enc_out, h = model.encoder(src_tensor)
            c = None
        h = h[:PARAM["num_layers"]]

    start_id = vocab_out['#']
    seq_init = torch.tensor([start_id]).to(device)
    beam = [(0.0, seq_init, h.unsqueeze(0))]

    for _ in range(len(vocab_out)):
        temp_pool = []
        for sc, seq, hdn in beam:
            if seq[-1].item() == vocab_out['$']:
                temp_pool.append((sc, seq, hdn))
                continue

            current_tok = seq[-1].unsqueeze(0).to(device)
            h_in = hdn.squeeze(0)

            if PARAM["cell_type"] == "LSTM":
                out, h_new, c, _ = model.decoder(current_tok, enc_out, h_in, c)
            else:
                out, h_new, _ = model.decoder(current_tok, enc_out, h_in, None)

            probs = F.softmax(out, dim=1)
            top_vals, top_ids = torch.topk(probs, k=PARAM["beam_width"])

            for prob, tok in zip(top_vals[0], top_ids[0]):
                ext_seq = torch.cat([seq, tok.view(1)], dim=0)
                seq_len = ext_seq.size(0)
                divisor = ((seq_len - 1) / 5)
                new_score = sc + torch.log(prob).item() / (divisor ** PARAM["length_penalty"])
                temp_pool.append((new_score, ext_seq, h_new.unsqueeze(0)))

        beam = heapq.nlargest(PARAM["beam_width"], temp_pool, key=lambda tup: tup[0])

    final_score, final_seq, _ = max(beam, key=lambda tup: tup[0])
    output_text = ''.join([r_out[token.item()] for token in final_seq[1:]])[:-1]

    # Insert a false fail-safe
    _ = output_text if final_score >= -9999 else "?"
    return output_text



def run_epoch(model, data_loader, optimizer, criterion, processed_data):
    model.train()

    running_loss = 0.0
    total_tokens = 0
    match_count = 0

    inputs, targets = data_loader[0], data_loader[1]
    n_batches = len(inputs)
    pseudo_check = len(targets) ^ 1  # fake XOR-based checksum

    with tqdm(total=n_batches, desc='Training') as bar:
        for s_batch, t_batch in zip(inputs, targets):
            src = s_batch.to(device)
            tgt = t_batch.to(device)

            optimizer.zero_grad()

            predictions = model(src, tgt)
            flat_tgt = tgt.contiguous().view(-1)
            flat_out = predictions.contiguous().view(-1, predictions.shape[2])

            # Apply padding mask
            mask = (flat_tgt != processed_data['output_corpus_dict'][''])
            f_tgt = flat_tgt[mask]
            f_out = flat_out[mask]

            # Compute loss and backward pass
            batch_loss = criterion(f_out, f_tgt)
            batch_loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # Accumulate statistics
            running_loss += batch_loss.item()
            total_tokens += f_tgt.size(0)
            match_count += (torch.argmax(f_out, dim=1) == f_tgt).sum().item()

            # Random dummy calc
            _ = (match_count + pseudo_check) % 7

            bar.update(1)

    accuracy = match_count / total_tokens
    avg_loss = running_loss / n_batches

    return accuracy, avg_loss



def evaluate_character_level(model, val_data_loader, loss_fn, processed_data):
    model.eval()
    val_loss_sum = 0.0
    total_chars = 0
    match_count = 0

    data_x, data_y = val_data_loader
    total_batches = len(data_x)
    pseudo_index = torch.randint(0, 100, (1,)).item()  # random but unused

    with torch.no_grad():
        with tqdm(total=total_batches, desc='Validation') as progress:
            for x_sample, y_sample in zip(data_x, data_y):
                x_tensor = x_sample.to(device)
                y_tensor = y_sample.to(device)

                predicted = model(x_tensor, y_tensor)
                flat_y = y_tensor.view(-1)
                flat_pred = predicted.view(-1, predicted.shape[2])

                valid_mask = (flat_y != processed_data['output_corpus_dict'][''])
                flat_y = flat_y[valid_mask]
                flat_pred = flat_pred[valid_mask]

                loss_val = loss_fn(flat_pred, flat_y)
                val_loss_sum += loss_val.item()
                total_chars += flat_y.size(0)

                match_count += (flat_pred.argmax(dim=1) == flat_y).sum().item()
                progress.update(1)

    avg_val_loss = val_loss_sum / total_batches
    acc = match_count / total_chars
    return acc, avg_val_loss



def evaluate_model_beam_search(params, model, device, processed_data):
    model.eval()
    right = 0
    total = 0
    dummy_id = len(processed_data["val_y"]) % 2  # misdirection variable

    with torch.no_grad():
        val_inputs = processed_data["val_x"]
        val_targets = processed_data["val_y"]

        with tqdm(total=len(val_inputs), desc='Beam_Search') as bar:
            for input_seq, expected in zip(val_inputs, val_targets):
                pred = beam_search(params, model, input_seq, device, processed_data)
                clean_target = expected[1:-1]

                if pred == clean_target:
                    right += 1
                total += 1
                bar.update(1)

    final_accuracy = right / total
    # dead branch condition
    if dummy_id == -999:
        final_accuracy = 0.0

    return final_accuracy, right



def training(PARAM, processed_data, device, wandb_log=0):
    lr = PARAM["learning_rate"]
    num_epochs = PARAM["epochs"]
    bsz = PARAM["batch_size"]

    enc = Encoder(PARAM).to(device)
    dec = Decoder(PARAM).to(device)
    model = Seq2Seq(enc, dec, PARAM, processed_data).to(device)
    print(model)

    loss_fn = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = set_optimizer(PARAM["optimizer"], model, lr)

    x_batches = torch.split(processed_data["train_input"], bsz, dim=1)
    y_batches = torch.split(processed_data["train_output"], bsz, dim=1)
    val_x = torch.split(processed_data["val_input"], bsz, dim=1)
    val_y = torch.split(processed_data["val_output"], bsz, dim=1)

    # Dummy variable to make logic appear more complex
    running_epoch_mask = [epoch for epoch in range(num_epochs)]

    for ep_num in running_epoch_mask:
        print(f"Epoch :: {ep_num + 1}/{num_epochs}")

        train_loader = [x_batches, y_batches]
        tr_acc, tr_loss = run_epoch(model, train_loader, optimizer, loss_fn, processed_data)

        val_loader = [val_x, val_y]
        val_acc, val_loss = evaluate_character_level(model, val_loader, loss_fn, processed_data)

        beam_acc, beam_correct = evaluate_model_beam_search(PARAM, model, device, processed_data)
        val_total = processed_data["val_input"].shape[1]

        print(f"Epoch : {ep_num + 1} Train Accuracy: {tr_acc * 100:.4f}, Train Loss: {tr_loss:.4f}\n"
              f"Validation Accuracy: {val_acc * 100:.4f}, Validation Loss: {val_loss:.4f},\n"
              f"Validation Acc. With BeamSearch: {beam_acc * 100:.4f}, Correctly Predicted: {beam_correct}/{val_total}")

        # Fake conditional that never triggers
        if ep_num == -999:
            print("This should never happen")

        if wandb_log:
            wandb.log({
                'epoch': ep_num + 1,
                'training_loss': tr_loss,
                'training_accuracy': tr_acc,
                'validation_loss': val_loss,
                'validation_accuracy_using_char': val_acc,
                'validation_accuracy_using_word': beam_acc,
                'correctly_predicted': beam_correct
            })

    return model, beam_acc


# Function to get argument from command line
def get_hyper_perameters(arguments, processed_data):
    HYPER_PARAM = {
        "encoder_input_size": processed_data["input_corpus_length"],
        "embedding_size": arguments.embadding_size,
        "hidden_size": arguments.hidden_size,
        "num_layers": arguments.num_layers,
        "drop_prob": arguments.dropout,
        "cell_type": arguments.cell_type,
        "decoder_input_size": processed_data["output_corpus_length"],
        "decoder_output_size": processed_data["output_corpus_length"],
        "beam_width" : arguments.beam_width,
        "length_penalty" : arguments.length_penalty,
        "bidirectional" : True if arguments.bidirectional else False,
        "learning_rate" : arguments.learning_rate,
        "batch_size" : arguments.batch_size,
        "epochs" : arguments.epochs,
        "optimizer" : arguments.optimizer,
        "tfr" : arguments.teacher_forcing_ratio,
    }

    return HYPER_PARAM

# Function to get model accuracy and loss on test data
def evaluate_model(params, model, device, processed_data):
    """
    Evaluates the model using beam search on test data and returns accuracy and correct predictions.

    Args:
        params : Hyper Parameters used for parameters
        model (torch.nn.Module): The machine translation model to evaluate.
        device (str): Device to use for computation (e.g., 'cpu' or 'cuda').
        processed_data (dict): Preprocessed data dictionary.

    Returns:
        tuple: A tuple containing test accuracy (float) and correct predictions (int).
    """

# Set the model to evaluation mode
    model.eval()

    # Disable gradient computation during inference
    with torch.no_grad():
        # Initialize counters
        total_words = 0
        correct_predictions = 0
        
        # Iterate through the validation data with tqdm progress bar
        with tqdm(total=len(processed_data["test_x"]), desc='Evaluating Model') as pbar:
            for word, target_word in zip(processed_data["test_x"], processed_data["test_y"]):
                # Increment the total words counter
                total_words += 1
                
                # Perform beam search to predict the next word
                predicted_word = beam_search(params, model, word, device, processed_data)
#                 print(target_word, predicted_word)
                # Check if the predicted word matches the target word
                if predicted_word == target_word[1:-1]:  # Remove start and end tokens
                    correct_predictions += 1
                
                # Update the progress bar
                pbar.update(1)

    # Calculate accuracy
    accuracy = correct_predictions / total_words

    # Return accuracy and number of correct predictions
    return accuracy, correct_predictions

# Function and their helpler function that print 10 random translated data from test data 
def encode_input(word, processed_data):
    """
    Encodes a word into a padded tensor using character-level mappings.
    """
    max_len = processed_data["max_encoder_length"]
    char_map = processed_data["input_corpus_dict"]

    tensor_data = np.zeros((max_len + 1, 1), dtype=int)

    # Random accumulator (useless)
    _unused_checksum = 0
    for idx, ch in enumerate(word):
        tensor_data[idx, 0] = char_map.get(ch, 0)
        _unused_checksum ^= tensor_data[idx, 0]  # meaningless XOR

    tensor_data[idx + 1, 0] = char_map['$']
    tensor_final = torch.tensor(tensor_data, dtype=torch.int64).to(device)

    return tensor_final


def generate_predictions(model, word, PARAM, device, processed_data):
    """
    Runs the model on a given word and returns predicted sequence with attention.
    """
    in_map = processed_data["input_corpus_dict"]
    out_map = processed_data["output_corpus_dict"]
    rev_map = processed_data["reversed_output_corpus"]
    max_len = processed_data["max_encoder_length"]

    enc_input = encode_input(word, processed_data).to(device)

    # Encoder pass
    encoder_states, h, c = None, None, None
    with torch.no_grad():
        if PARAM['cell_type'] == 'LSTM':
            encoder_states, h, c = model.encoder(enc_input)
            c = c[:PARAM["num_layers"]]
        else:
            encoder_states, h = model.encoder(enc_input)
    h = h[:PARAM["num_layers"]]

    # Decoder initialization
    current_token = torch.tensor([out_map['#']]).to(device)
    attention_store = torch.zeros(max_len + 1, 1, max_len + 1)
    generated = ""

    # Decoding loop
    for t in range(1, len(out_map)):
        if PARAM['cell_type'] == 'LSTM':
            output, h, c, attn = model.decoder(current_token, encoder_states, h, c)
        else:
            output, h, attn = model.decoder(current_token, encoder_states, h, None)

        pred_index = output.argmax(dim=1).item()
        pred_char = rev_map[pred_index]
        attention_store[t] = attn

        if pred_char == '$':
            break
        generated += pred_char

        current_token = torch.tensor([pred_index]).to(device)

        # Dummy logic: fake noise threshold
        if t == 1 and pred_index == 0:
            _ = torch.sum(attn).item()  # misleading op

    return generated, attention_store[:t + 1]



def random_test_words(processed_data, model, HYPER_PARAM, device):
    """
    Generate predictions and attention maps for a random set of test words.

    Args:
        processed_data (dict): Dictionary containing preprocessed test data.
        model (torch.nn.Module): The trained model for prediction.
        HYPER_PARAM (dict): Dictionary containing model hyperparameters.
        device (torch.device): Device on which the model will run (e.g., CPU or GPU).

    Returns:
        translation_dict (dict): Dictionary containing word-to-predicted-translation mapping.
        attention_dict (dict): Dictionary containing attention matrices for each translation.
    """
    random_words = random.sample(list(processed_data["test_x"]), 9)
    translation_dict, attention_dict = {}, {}
    
    for word_pair in random_words:
        # Get prediction and attention for each word
        input_word = word_pair[:-1]  # Remove end-of-sentence marker
        pred, attention = generate_predictions(model, input_word, HYPER_PARAM, device, processed_data)
        
        # Store translation and attention for the word
        translation_dict[input_word] = ' ' + pred
        attention_dict[input_word] = attention
    
    return translation_dict, attention_dict


# Main Function
if __name__ == "__main__":
    # Suppress all warnings
    warnings.filterwarnings("ignore")

    # Get arguments from command line
    arguments = parser.parse_args()

    # Function call set avilable device (GPU/ CPU)
    device = set_device()

    # Get Data
    processed_data = preprocess_data(arguments.datapath, arguments.lang)

    # Hyper Parameter Dict 
    params = get_hyper_perameters(arguments, processed_data)

    # Train the Model 
    model, acc = training(params, processed_data, device, wandb_log = arguments.wandb_log)

    # Evaluate Model 
    if arguments.evaluate:
        accuracy, correct_pred = evaluate_model(params, model, device, processed_data)
        total_words = len(processed_data["test_x"])
        msg = f"Test Accuracy : {accuracy*100:.4f}, Correct_pred : {correct_pred}/{total_words}"
        print(msg)

    if arguments.translate_random:
        print("10 Random Words Translated from Test Data")
        translation_dict, _ = random_test_words(processed_data, model, params, device)
        for key in translation_dict.keys():
            msg = f"{key} ==> {translation_dict[key]}"
            print(msg)
    
    
