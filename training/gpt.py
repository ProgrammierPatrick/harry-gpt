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
import os, math, time, datetime

LANGUAGE = "de"

TRAIN_DATA_FILE = f"../data/HP1-7_{LANGUAGE}.txt"
OUTPUT_FOLDER = "result"

NUM_LAYERS = 1
EMBED_FEATURES  = 16
HIDDEN_FEATURES = 64
NHEAD = 2
TEMPERATURE = 0.7

BATCH_SIZE = 64
SEQUENCE_LENGTH = 128 # 512
LEARNING_RATE = 0.1
LEARNING_BETA = (0.8, 0.999)
NUM_EPOCHS = 4

MODEL_NAME = f"gpt-{HIDDEN_FEATURES}-{LANGUAGE}"

# DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DEVICE = 'cuda'
# DEVICE = 'cpu'

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
data_ix = torch.tensor(data_ix).to(DEVICE)
print('done.')

class PositionalEncoding(nn.Module):
  def __init__(self, d_model: int, max_len: int=5000, dropout: float=0.1):
    super().__init__()
    self.dropout = nn.Dropout(p=dropout)
    pe = torch.zeros(max_len, d_model)  # like 10x4
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    self.register_buffer('pe', pe)
    # pe (max_len, d_model)
  def forward(self, x):
    # x (batch_size, seq_len, d_model)
    x = x + self.pe[:x.size(1)]
    return self.dropout(x)
  
class LearnablePosEnc(nn.Module):
    def __init__(self, d_model: int, max_len: int=5000, dropout: float=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.embedding = nn.Embedding(num_embeddings=max_len, embedding_dim=d_model)
    def forward(self, x):
        # x (batch_size, seq_len, d_model)
        seq_len = x.size(1)
        x = x + self.embedding(torch.arange(seq_len, device=x.device))
        return self.dropout(x)

class Model(nn.Module):
    def __init__(self, vocab_size, sequence_length, hidden_features, nhead, num_layers, temperature, char_to_ix, ix_to_char):
        super().__init__()
        self.vocab_size = vocab_size
        self.sequence_length = sequence_length
        self.hidden_features = hidden_features
        self.nhead = nhead
        self.num_layers = num_layers
        self.temperature = temperature
        self.char_to_ix = char_to_ix
        self.ix_to_char = ix_to_char

        self.info_name: str = MODEL_NAME
        self.info_epochs: float = 0.0
        self.info_time: float = 0.0
        self.info_loss: float = float('inf')
        self.info_batch_size: int = BATCH_SIZE
        self.date: str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.is_snapshot: bool = True

        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=hidden_features)
        # self.pos_encoder = PositionalEncoding(d_model=hidden_features, max_len=sequence_length, dropout=0.1)
        self.pos_encoder = LearnablePosEnc(d_model=hidden_features, max_len=sequence_length, dropout=0.1)

        self.transformer_layer = nn.TransformerEncoderLayer(d_model=hidden_features, nhead=nhead, dim_feedforward=2*hidden_features, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer=self.transformer_layer, num_layers=self.num_layers)
        src_mask = nn.Transformer.generate_square_subsequent_mask(sequence_length)
        self.register_buffer('src_mask', src_mask)

        self.out_embed = nn.Linear(hidden_features, vocab_size)

    def forward(self, input: Tensor) -> Tensor:
        # input: (batch, seq_len)
        batch_size = input.size(0)
        seq_len = input.size(1)

        # pad zeros to the right
        value = input[:, max(0, seq_len - self.sequence_length):]
        value = torch.cat((value, torch.zeros(batch_size, max(0, self.sequence_length - seq_len), dtype=torch.long, device=input.device)), dim=1)
        # value: (batch, max_seq_len)

        value = self.embedding(value)
        # value: (batch, max_seq_len, hidden_features)

        value = self.pos_encoder(value)
        # value: (batch, max_seq_len, hidden_features)

        value = torch.cat((torch.zeros(batch_size, 1, self.hidden_features, device=input.device), value[:, :-1]), dim=1)
        # value: (batch, max_seq_len, hidden_features)

        value = self.transformer(src=value, mask=self.src_mask)

        # value: (batch, max_seq_len, hidden_features)

        value = self.out_embed(value)
        # value: (batch, max_seq_len, vocab_size)

        value = value[:, :seq_len]
        # value: (batch, seq_len, vocab_size)

        return value

    @torch.jit.export
    def generate_start(self, prompt: str = "") -> Tuple[str, List[Tensor]]:
        device = self.out_embed.weight.device

        prompt_ix: List[int] = []
        for c in prompt: prompt_ix.append(self.char_to_ix[c])
        # prompt_ix: List(len(prompt),)

        prompt_value = torch.tensor(prompt_ix, device=device).view(1, -1)
        # prompt_value: (1, len(prompt))

        return self.generate_step([prompt_value])

    @torch.jit.export
    def generate_step(self, context: List[Tensor]) -> Tuple[str, List[Tensor]]:
        with torch.no_grad():
            prompt_value = context[0]
            # value: (1, len(prompt))

            # add <END> token (ignored by right shifting)
            prompt_value = torch.cat((prompt_value, torch.zeros(1, 1, dtype=torch.long, device=prompt_value.device)), dim=1)
            # value: (1, len(prompt)+1)

            result_value = self(prompt_value)
            # result_value: (1, len(prompt)+1, vocab_size)

            result_value = result_value[:,-1].ravel()
            # result_value: (vocab_size,)

            result_value = F.softmax(result_value / self.temperature, dim=-1)
            result_value = torch.multinomial(result_value, num_samples=1)
            # result_value: (1,)

            c = self.ix_to_char[int(result_value)]

            prompt_value[-1] = result_value
            # prompt_value: (1, len(prompt)+1)

            return c, [prompt_value]


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


model = Model(vocab_size=vocab_size, sequence_length=SEQUENCE_LENGTH, hidden_features=HIDDEN_FEATURES, nhead=NHEAD,
              num_layers=NUM_LAYERS, temperature=TEMPERATURE, char_to_ix=char_to_ix, ix_to_char=ix_to_char)
model.to(DEVICE)
loss_func = nn.CrossEntropyLoss().to(DEVICE)
# optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, betas=LEARNING_BETA)
optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=LEARNING_BETA[0])


fig, ax = plt.subplots(1,1)
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
    ax.set_ylim(bottom=0)
    ax.grid('on')
    fig.canvas.draw_idle()
    fig.canvas.flush_events()
    fig.show()

def save_model(is_snapshot: bool = False, trained_epochs: float = 0):
    print("saving model...")
    update_plot()

    name = MODEL_NAME + ("_snapshot" if is_snapshot else "")

    last_losses = loss_graph[-10:]
    avg_loss = sum(last_losses) / len(last_losses)

    model.info_epochs = trained_epochs if is_snapshot else float(NUM_EPOCHS)
    model.info_time = time.time() - start_time
    model.info_loss = avg_loss
    model.info_is_snapshot = is_snapshot

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    scripted_model = torch.jit.script(model)
    scripted_model.save(os.path.join(OUTPUT_FOLDER, name + ".pt"))

    plt.savefig(os.path.join(OUTPUT_FOLDER, name + ".png"))
    print("model saved.")

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
generate_timer = Timer(interval_in_s=30) # 30s
snapshot_timer = Timer(interval_in_s=60) # 60s

for epoch_idx in range(NUM_EPOCHS):
    # epoch_offset = torch.randint(0,SEQUENCE_LENGTH-1, (BATCH_SIZE,)) # BATCH_SIZE offsets
    for batch_idx in range((data_size // SEQUENCE_LENGTH) - 1):
        this_batch_time = time.time()

        offset = torch.randint(0, data_size - SEQUENCE_LENGTH - 1, (BATCH_SIZE,))

        # indices_expanded = epoch_offset[:, None].expand(-1, SEQUENCE_LENGTH + 1)
        indices_expanded = offset[:, None].expand(-1, SEQUENCE_LENGTH + 1)
        offsets = torch.arange(SEQUENCE_LENGTH + 1).expand(offset.size(0), -1)
        indices_2d = indices_expanded + offsets
        data_slice = data_ix[indices_2d]
        data_input_slice = data_slice[:,:-1].contiguous()
        data_output_slice = data_slice[:,1:].contiguous()
        # data_*_slice: (batch, seq_len)

        # GENERATE TEXT
        if generate_timer.check(this_batch_time):
            print (f"batch {batch_idx}/{data_size // SEQUENCE_LENGTH}")
            print("----")
            print(generate(model=model, length=2000))
            print("----")


        result = model.forward(data_input_slice)
        target = F.one_hot(data_input_slice, num_classes=vocab_size).float()
        loss = loss_func(result.view(-1, vocab_size), target.view(-1, vocab_size))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_graph.append(float(loss))
        loss_x.append(epoch_idx + batch_idx / (data_size // SEQUENCE_LENGTH))
        loss_time.append(this_batch_time - start_time)

        # PLOT LOSS
        if plot_timer.check(this_batch_time):
            print(f"epoch {epoch_idx} / {NUM_EPOCHS} batch {batch_idx} / {(data_size - 1) // SEQUENCE_LENGTH}: loss: {loss}")
            update_plot()

        if snapshot_timer.check(this_batch_time):
            save_model(is_snapshot=True, trained_epochs=epoch_idx + batch_idx / (data_size // SEQUENCE_LENGTH))

print("TRAINING COMPLETED!!!")
print("GENERATE NAME...")
book_name = f"Harry Potter and {generate(model=model, length=100, prompt='Harry Potter and ')}"
print(book_name)

print ("GENERATE TEXT...")
book_text = generate(model=model, length=10_000)
print(book_text)

# torch.save(model.state_dict(), 'hp_gan.pt')
# with open('harry_new_book.txt', 'w') as f:
#     f.write(book_name)
#     f.write("")
#     f.write(book_text)

save_model()