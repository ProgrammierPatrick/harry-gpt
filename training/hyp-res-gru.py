#!/usr/bin/env python3
import torch
import torch.utils
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.utils.rnn as rnn
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from typing import Dict, Iterator, List, Tuple, Union
from torch import Tensor
import os, time, datetime
import optuna

LANGUAGE = "de"

TRAIN_DATA_FILE = f"../data/HP1-7_{LANGUAGE}.txt"
OUTPUT_FOLDER = "result"
STUDY_DB = "sqlite:///result/optuna.db"

MINUTES_PER_MODEL = 2

EMBED_FEATURES  = 32
HIDDEN_FEATURES = 256
NUM_LAYERS = 2
DROPOUT = 0.0
TEMPERATURE = 0.7

BATCH_SIZE = 128
SEQUENCE_LENGTH = 512 # 128
LEARNING_RATE = 0.001
LEARNING_BETA = (0.7, 0.999)
WEIGHT_DECAY = 0.01
GRADIENT_CLIP = 5.0
NUM_EPOCHS = 1

STUDY_NAME = f"hyp-gru-res-{NUM_LAYERS}-{MINUTES_PER_MODEL}min-{LANGUAGE}"
MODEL_NAME = STUDY_NAME

data = open(TRAIN_DATA_FILE, 'r').read() # should be simple plain text file
chars = list(set(data))
data_size, vocab_size = len(data), len(chars)
print('data has %d characters, %d unique.' % (data_size, vocab_size))
char_to_ix = { ch:i for i,ch in enumerate(chars) }
ix_to_char = { i:ch for i,ch in enumerate(chars) }

print("vocabulary:", chars)
print("ix_to_char:", ix_to_char)

data_ix = list(map(lambda c: char_to_ix[c], data))
print('upload training data...')
data_ix = torch.tensor(data_ix, dtype=torch.long).cuda()
print('done.')

class ResidualGruBlock(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.lin = nn.Sequential(
            nn.Linear(hidden_size, 2 * hidden_size),
            nn.GELU(),
            nn.Linear(2 * hidden_size, hidden_size))
    
    def forward(self, x: Tensor, h: Tensor) -> Tuple[Tensor, Tensor]:
        val, h = self.gru(x, h)
        val = self.lin(val)
        return x + val, h

class ResidualGruStack(nn.Module):
    def __init__(self, num_layers: int, input_size: int, hidden_features: int, dropout: float = 0.0):
        super().__init__()
        self.first_gru = nn.GRU(input_size=input_size, hidden_size=hidden_features, batch_first=True)
        self.first_lin = nn.Sequential(
            nn.Linear(hidden_features, 2 * hidden_features),
            nn.GELU(),
            nn.Linear(2 * hidden_features, hidden_features))
        self.other_grus = nn.ModuleList([ResidualGruBlock(input_size=hidden_features, hidden_size=hidden_features) for _ in range(num_layers - 1)])

    def forward(self, x: Tensor, h: Tensor) -> Tuple[Tensor, Tensor]:
        x, h_1 = self.first_gru(x, h[:1])
        x = self.first_lin(x)
        hs: List[Tensor] = [h_1]
        for i, g in enumerate(self.other_grus):
            x, hi = g(x, h[i+1:i+2])
            hs.append(hi)
        return x, torch.cat(hs)

class Model(nn.Module):
    def __init__(self, vocab_size, embed_features, hidden_features, num_layers, dropout, temperature, char_to_ix, ix_to_char):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_features = embed_features
        self.hidden_features = hidden_features
        self.num_layers = num_layers
        self.dropout = dropout
        self.char_to_ix = char_to_ix
        self.ix_to_char = ix_to_char
        self.temperature = temperature

        self.info_name: str = MODEL_NAME
        self.info_epochs: float = 0.0
        self.info_time: float = 0.0
        self.info_loss: float = float('inf')
        self.info_batch_size: int = BATCH_SIZE
        self.date: str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.is_snapshot: bool = True

        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_features)
        self.gru = ResidualGruStack(num_layers=num_layers, input_size=embed_features, hidden_features=hidden_features, dropout=dropout)
        self.out_embed = nn.Linear(hidden_features, vocab_size)

    # input: (batch, seq_len)
    def forward(self, input: Tensor, memory: Tensor) -> Tuple[Tensor, Tensor]:
        embedded = self.embedding(input)
        # embedded: (batch, seq_len, embed_features)
        if memory is None:
            memory = torch.zeros(self.num_layers, input.size(0), self.hidden_features, device=input.device)
        hidden, hn = self.gru(embedded, memory)

        # hidden: (batch, seq_len, hidden_features)
        # hn, cn: (seq_len, batch, hidden_features)
        output_embed = self.out_embed(hidden)
        # output_embed: (batch, seq_len, vocab_size)
        return output_embed, hn.detach()

    @torch.jit.export
    def generate_start(self, prompt: str = "") -> Tuple[str, List[Tensor]]:
        device = self.out_embed.weight.device

        hn = torch.zeros(self.num_layers, 1, self.hidden_features, device=device)

        prompt_ix: List[int] = []
        for c in prompt: prompt_ix.append(self.char_to_ix[c])
        # prompt_ix: List(seq_len=len(prompt),)

        value, hn = self(torch.tensor(prompt_ix, device=device).view(1, -1), hn)
        # value: (1, len(prompt), vocab_size)
        value = F.softmax(value[:,-1].ravel() / self.temperature, dim=-1)
        # value: (vocab_size,)
        value = torch.multinomial(value, num_samples=1)
        # value: (1,)

        c = self.ix_to_char[int(value)]
        return c, [value, hn]

    @torch.jit.export
    def generate_step(self, context: List[Tensor]) -> Tuple[str, List[Tensor]]:
        value, hn = context

        value, hn = self(value.view(1, -1), hn)
        # value: (1, 1, vocab_size)
        value = F.softmax(value[:,-1].ravel() / self.temperature, dim=-1)
        # value: (vocab_size,)
        value = torch.multinomial(value, num_samples=1)
        # value: (1,)

        c = self.ix_to_char[int(value)]
        return c, [value, hn]


def generate_it(model: nn.Module, length: int, prompt: str = "") -> Iterator[str]:
    if prompt == "":
        prompt = model.ix_to_char[torch.randint(low=0, high=model.vocab_size, size=(1,)).item()]
    char, context = model.generate_start(prompt)
    yield char
    for _ in range(length - 1):
        char, context = model.generate_step(context)
        yield char

def generate(model: nn.Module, length: int, prompt: str = "") -> str:
    return "".join(generate_it(model=model, length=length, prompt=prompt))

fig, ax = plt.subplots(1,1)

def objective(trial):
    embed_features = trial.suggest_int("embed_features", 4, 256, log=True)
    hidden_features = trial.suggest_int("hidden_features", 32, 1024, log=True)
    dropout = trial.suggest_float("dropout", 0.0, 0.8)

    batch_size = trial.suggest_categorical("batch_size", [1, 2, 4, 8, 16, 32, 64, 128, 256, 512])
    seq_length = trial.suggest_categorical("seq_length", [16, 32, 64, 128, 256, 512, 1024, 2048, 4096])
    gradient_clip = trial.suggest_float("gradient_clip", 1.0, 100.0, log=True)

    model = Model(vocab_size=vocab_size, embed_features=embed_features,
                hidden_features=hidden_features, num_layers=NUM_LAYERS,
                dropout=dropout,
                temperature=TEMPERATURE, char_to_ix=char_to_ix, ix_to_char=ix_to_char)
    model.cuda()
    loss_func = nn.CrossEntropyLoss().cuda()

    optimizer_type = trial.suggest_categorical("optimizer", ["Adam", "AdamW", "RMSprop", "SGD"])
    match optimizer_type:
        case "Adam" | "AdamW":
            optimizer_params = {
                "type": optimizer_type,
                "lr": trial.suggest_float("lr", 1e-7, 1e-1, log=True),
                "betas": (trial.suggest_float("beta1", 0.0, 1.0), trial.suggest_float("beta2", 0.0, 1.0)),
                "weight_decay": trial.suggest_float("weight_decay", 0.0, 1e-1),
            }
        case "RMSprop":
            optimizer_params = {
                "type": optimizer_type,
                "lr": trial.suggest_float("lr", 1e-9, 1e+2, log=True),
                "alpha": trial.suggest_float("alpha", 0.0, 1.0),
                "eps": trial.suggest_float("eps", 1e-9, 1e-1, log=True),
                "weight_decay": trial.suggest_float("weight_decay", 0.0, 1e-1),
            }
        case "SGD":
            optimizer_params = {
                "type": optimizer_type,
                "lr": trial.suggest_float("lr", 1e-9, 1e+2, log=True),
                "momentum": trial.suggest_float("momentum", 0.0, 1.0),
                "weight_decay": trial.suggest_float("weight_decay", 0.0, 1e-1),
            }
    
    def create_optimizer(opt_type):
        match opt_type["type"]:
            case "Adam":  return optim.Adam(model.parameters(), lr=opt_type["lr"], betas=opt_type["betas"], weight_decay=opt_type["weight_decay"])
            case "AdamW": return optim.AdamW(model.parameters(), lr=opt_type["lr"], betas=opt_type["betas"], weight_decay=opt_type["weight_decay"])
            case "RMSprop": return optim.RMSprop(model.parameters(), lr=opt_type["lr"], alpha=opt_type["alpha"], eps=opt_type["eps"], weight_decay=opt_type["weight_decay"])
            case "SGD": return optim.SGD(model.parameters(), lr=opt_type["lr"], momentum=opt_type["momentum"], weight_decay=opt_type["weight_decay"])

    optimizer = create_optimizer(optimizer_params)

    loss_graph = []
    loss_x = []
    loss_time = []

    start_time = time.time()

    def update_plot():
        ax.clear()
        ax.plot(loss_x, loss_graph, label="loss")
        ax.set_title(f"Model '{MODEL_NAME}': Cross Entropy Loss")
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Loss")
        if loss_graph[-1] > 4:
            ax.set_ylim(top=None)
        elif loss_graph[-1] > 1:
            ax.set_ylim(top=4)
        else:
            ax.set_ylim(top=2)
        ax.set_ylim(bottom=0)
        ax.grid('on')
        fig.canvas.draw_idle()
        fig.canvas.flush_events()
        fig.show()

    class Timer:
        def __init__(self, interval_in_s: float):
            self.interval_in_s = interval_in_s
            self.last_time = time.time()
        def check(self, now: float) -> bool:
            if now - self.last_time > self.interval_in_s:
                self.last_time = now
                return True
            return False

    plot_timer = Timer(interval_in_s=1)
    generate_timer = Timer(interval_in_s=30)

    min_norm = 1e10
    clip_count = 0

    for epoch_idx in range(NUM_EPOCHS):
        epoch_memory = None
        for batch_idx in range((data_size // seq_length) - 1):
            this_batch_time = time.time()

            epoch_memory = None
            offset = torch.randint(0, data_size - seq_length - 1, (batch_size,))

            # indices_expanded = epoch_offset[:, None].expand(-1, SEQUENCE_LENGTH + 1)
            indices_expanded = offset[:, None].expand(-1, seq_length + 1)
            offsets = torch.arange(seq_length + 1).expand(offset.size(0), -1)
            indices_2d = indices_expanded + offsets
            data_slice = data_ix[indices_2d]
            data_input_slice = data_slice[:,:-1].contiguous()
            data_output_slice = data_slice[:,1:].contiguous()
            # data_*_slice: (batch, seq_len)

            # GENERATE TEXT
            if generate_timer.check(this_batch_time):
                print (f"batch {batch_idx}/{data_size // seq_length}")
                print("----")
                for c in generate_it(model=model, length=500):
                    print(c, end="")
                print("\n----")


            result, epoch_memory = model.forward(data_input_slice, epoch_memory)
            target = F.one_hot(data_output_slice, num_classes=vocab_size).float()
            loss = loss_func(result.view(-1, vocab_size), target.view(-1, vocab_size))

            optimizer.zero_grad()
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            
            min_norm = min(min_norm, norm)
            if norm > gradient_clip and min_norm < gradient_clip / 2 and (epoch_idx > 0 or batch_idx > 20):
                print("reset optimizer")
                optimizer = create_optimizer(optimizer_params)
                clip_count += 1
            optimizer.step()

            loss_graph.append(float(loss))
            loss_x.append(epoch_idx + batch_idx / (data_size // seq_length))
            loss_time.append(this_batch_time - start_time)

            # PLOT LOSS
            if plot_timer.check(this_batch_time):
                print(f"epoch {epoch_idx} / {NUM_EPOCHS} batch {batch_idx} / {(data_size - 1) // seq_length}: loss: {loss} grad: {norm}")
                update_plot()
            
            trial.report(loss, epoch_idx + batch_idx / (data_size // seq_length))
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned("Pruned by pruner")

            can_prune = epoch_idx > 1 or batch_idx > 50

            if can_prune and loss > 9000:
                raise optuna.exceptions.TrialPruned("Loss oer 9000")
            
            if loss == float("nan") or loss == float("inf") or loss == float("-inf"):
                raise optuna.exceptions.TrialPruned("Loss is nan / inf / -inf")

            if can_prune and clip_count > (epoch_idx * (data_size // seq_length) + batch_idx) / 10:
                raise optuna.exceptions.TrialPruned("Too many clips")

            now = time.time()
            if now - start_time > 60 * MINUTES_PER_MODEL:
                return loss

    return loss    
    # last_losses = loss_graph[-10:]
    # avg_loss = sum(last_losses) / len(last_losses)
    # return avg_loss

if __name__ == "__main__":
    study = optuna.create_study(storage="sqlite:///example.db", study_name=STUDY_NAME, load_if_exists=True,
                                direction="minimize", pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=25))
    study.optimize(objective, n_trials=100, timeout=60 * 10, gc_after_trial=True)

    print(f"Best trial: {study.best_trial.value}")
    print("Best params:")
    for key, value in study.best_trial.params.items():
        print(f"    {key}: {value}")
    
