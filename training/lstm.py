import torch
import torch.utils
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.utils.rnn as rnn
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from typing import Dict, Tuple, Union
import os, time, datetime

LANGUAGE = "de"

TRAIN_DATA_FILE = f"../data/HP1-7_{LANGUAGE}.txt"
OUTPUT_FOLDER = "result"

# EMBED_FEATURES  = 16
EMBED_FEATURES  = 64
# HIDDEN_FEATURES = 512
HIDDEN_FEATURES = 128
# NUM_LAYERS = 4
NUM_LAYERS = 1

# BATCH_SIZE = 4
BATCH_SIZE = 16
# SEQUENCE_LENGTH = 2048
SEQUENCE_LENGTH = 1024
LEARNING_RATE = 0.002
LEARNING_BETA = (0.7, 0.999)
NUM_EPOCHS = 1

MODEL_NAME = f"lstm-{NUM_LAYERS}-{HIDDEN_FEATURES}-{LANGUAGE}"

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
data_ix = torch.tensor(data_ix).cuda()
print('done.')

class Model(nn.Module):
    def __init__(self, vocab_size, embed_features, hidden_features, num_layers, char_to_ix, ix_to_char):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_features = embed_features
        self.hidden_features = hidden_features
        self.num_layers = num_layers
        self.char_to_ix = char_to_ix
        self.ix_to_char = ix_to_char

        self.info_name: str = MODEL_NAME
        self.info_epochs: float = 0.0
        self.info_time: float = 0.0
        self.info_loss: float = float('inf')
        self.info_batch_size: int = BATCH_SIZE
        self.date: str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.is_snapshot: bool = True

        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_features)
        self.lstm = nn.LSTM(input_size=embed_features, hidden_size=hidden_features, num_layers=NUM_LAYERS, batch_first=True, dropout=0.2)
        self.out_embed = nn.Linear(hidden_features, vocab_size)

    # input: (batch, seq_len)
    def forward(self, input: torch.Tensor, memory: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        batch_size = input.size(0)
        embedded = self.embedding(input)
        # embedded: (batch, seq_len, embed_features)
        if memory is None:
            memory = (
                torch.zeros(self.num_layers, input.size(0), self.hidden_features, device=input.device),
                torch.zeros(self.num_layers, input.size(0), self.hidden_features, device=input.device)
            )
        hidden, (hn, cn) = self.lstm(embedded, memory)
        # hidden: (batch, seq_len, hidden_features)
        # hn, cn: (seq_len, batch, hidden_features)
        output_embed = self.out_embed(hidden)
        # output_embed: (batch, seq_len, vocab_size)
        return output_embed, (hn.detach(), cn.detach())
    
def generate(model: nn.Module, device: str, length: int, prompt: str = "") -> str:
    memory = None
    if len(prompt) == 0:
        value = torch.randint(0, model.vocab_size - 1, (1,1), device=device)
        # value: (batch=1,seq_len=1)
    else:
        prompt_ix = list(map(lambda c: model.char_to_ix[c], prompt))
        # prompt_ix: list(seq_len=len(prompt),)
        value, memory = model(torch.tensor(prompt_ix, device=device).view(1, -1), memory)
        # value: (batch=1, seq_len=len(prompt), vocab_size)
        value = F.softmax(value[:,-1].ravel(), dim=-1)
        # value: (vocab_size,)
        value = torch.multinomial(value.ravel(), 1)
        # value: (1,)
        value = value.view(1, 1)
        # value: (batch=1, seq_len=1)
    text = []
    for _ in range(length):
        value, memory = model(value, memory)
        # value: (batch=1, seq_len=1, vocab_size)
        value = F.softmax(value, dim=-1)
        # value: (batch=1, seq_len=1, vocab_size)
        value = torch.multinomial(value.ravel(), 1)
        # value: (1,)
        value = value.view(1, 1)
        # value: (batch=1, seq_len=1)
        text.append(ix_to_char[int(value)])
    return ''.join(text)

model = Model(vocab_size=vocab_size, embed_features=EMBED_FEATURES,
              hidden_features=HIDDEN_FEATURES, num_layers=NUM_LAYERS, char_to_ix=char_to_ix, ix_to_char=ix_to_char)
model.cuda()
loss_func = nn.CrossEntropyLoss().cuda()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, betas=LEARNING_BETA)

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
generate_timer = Timer(interval_in_s=30)
# snapshot_timer = Timer(interval_in_s=5*60)
snapshot_timer = Timer(interval_in_s=10)

for epoch_idx in range(NUM_EPOCHS):
    epoch_memory = None
    epoch_offset = torch.randint(0,SEQUENCE_LENGTH-1, (BATCH_SIZE,)) # BATCH_SIZE offsets
    for batch_idx in range((data_size // SEQUENCE_LENGTH) - 1):
        this_batch_time = time.time()

        indices_expanded = epoch_offset[:, None].expand(-1, SEQUENCE_LENGTH + 1)
        offsets = torch.arange(SEQUENCE_LENGTH + 1).expand(epoch_offset.size(0), -1)
        indices_2d = indices_expanded + offsets + batch_idx * SEQUENCE_LENGTH
        data_slice = data_ix[indices_2d]
        data_input_slice = data_slice[:,:-1].contiguous()
        data_output_slice = data_slice[:,1:].contiguous()
        # data_*_slice: (batch, seq_len)

        # GENERATE TEXT
        if generate_timer.check(this_batch_time):
            print (f"batch {batch_idx}/{data_size // SEQUENCE_LENGTH}")
            txt = generate(model, 'cuda', 2000)
            print (f'----\n {txt} \n----')

        result, epoch_memory = model.forward(data_input_slice, epoch_memory)
        target = F.one_hot(data_output_slice, num_classes=vocab_size).float()
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
book_name = f"Harry Potter and {generate(model, 'cuda', 100, 'Harry Potter and ')}"
print(book_name)

print ("GENERATE TEXT...")
book_text = generate(model, 'cuda', 10_000)
print(book_text)

# torch.save(model.state_dict(), 'hp_gan.pt')
# with open('harry_new_book.txt', 'w') as f:
#     f.write(book_name)
#     f.write("")
#     f.write(book_text)

save_model()