#!/usr/bin/env python3
from flask import Flask, render_template, request, Response, session, copy_current_request_context
import urllib.parse
import torch
import torch.nn.functional as F
import time
import secrets
import uuid
import os


app = Flask(__name__)
app.secret_key = secrets.token_hex(16)

models = {} # key: name, value: {model, [{"type", **params}]}

for model_filename in os.listdir("models"):
    model_name, _ = os.path.splitext(model_filename)
    path = os.path.join("models", model_filename)
    model = torch.jit.load(path, "cpu")

    layers = []
    for name, layer in model.named_children():
        if layer.original_name == "Linear":
            layers.append({"type": "Linear", "in_features": layer.in_features, "out_features": layer.out_features})
        elif layer.original_name == "LSTM":
            layers.append({"type": "LSTM", "input_size": layer.input_size, "hidden_size": layer.hidden_size, "num_layers": layer.num_layers, "dropout": layer.dropout})
        else:
            print(f"unknown layer type '{layer.original_name}' in file {model_filename}")

    models[model_name] = {"model": model, "layers": layers}

# model = models["hp_gan_8"]["model"]
model = models["hp_gan_full_test_next"]["model"]

active_generators = {}

def generate(length, prompt=""):
    memory = (torch.zeros(model.num_layers, 1, model.hidden_features),
              torch.zeros(model.num_layers, 1, model.hidden_features))
    if len(prompt) == 0:
        seed = torch.randint(0, model.vocab_size - 1, (1, 1))
        value = F.one_hot(seed, num_classes=model.vocab_size).float()
        value = value.view(1, 1, model.vocab_size)
    else:
        prompt_ix = list(map(lambda c: model.char_to_ix[c], prompt))
        prompt_onehot = F.one_hot(torch.tensor(prompt_ix), num_classes=model.vocab_size).float()
        value, memory = model.forward(prompt_onehot.view(1, len(prompt), model.vocab_size), memory)
        value = value[:,-1,:].view(1, 1, model.vocab_size)

    text = []
    for i in range(length):
        value, memory = model.forward(value, memory)
        value = F.softmax(value, dim=-1)
        ix = int(torch.multinomial(value.ravel(), 1).cpu()[0].detach())
        value = F.one_hot(torch.tensor([ix]), num_classes=model.vocab_size).float()
        text.append(model.ix_to_char[ix])
    return ''.join(text)

def generate_iter(length, prompt=""):
    memory = (torch.zeros(model.num_layers, 1, model.hidden_features),
              torch.zeros(model.num_layers, 1, model.hidden_features))
    if len(prompt) == 0:
        seed = torch.randint(0, model.vocab_size - 1, (1, 1))
        value = F.one_hot(seed, num_classes=model.vocab_size).float()
        value = value.view(1, 1, model.vocab_size)
    else:
        prompt_ix = list(map(lambda c: model.char_to_ix[c], prompt))
        prompt_onehot = F.one_hot(torch.tensor(prompt_ix), num_classes=model.vocab_size).float()
        value, memory = model.forward(prompt_onehot.view(1, len(prompt), model.vocab_size), memory)
        value = value[:,-1,:].view(1, 1, model.vocab_size)

    for i in range(length):
        value, memory = model.forward(value, memory)
        value = F.softmax(value, dim=-1)
        ix = int(torch.multinomial(value.ravel(), 1).cpu()[0].detach())
        value = F.one_hot(torch.tensor([ix]), num_classes=model.vocab_size).float()
        yield model.ix_to_char[ix]

@app.route('/sse/send_chat')
def sse_send_chat():
    if not 'uuid' in session:
        session_uuid = str(uuid.uuid4())
        session['uuid'] = session_uuid
        print("generated uuid:", session_uuid)
        session.modified = True
    else:
        session_uuid = session['uuid']
        print("keep old uuid:", session_uuid)
        
    @copy_current_request_context
    def outer_gen(uuid):
        try:
            while True:
                if uuid in active_generators:
                    yield f"data: {next(active_generators[uuid])}\n\n"
                else:
                    time.sleep(0.1)
        except StopIteration:
            pass
        print("Generation finished.")
    return Response(outer_gen(session_uuid), mimetype='text/event-stream')

@app.route('/ajax/send_chat', methods = ['POST'])
def ajax_send_chat():
    msg = request.form["msg"]
    msg = urllib.parse.unquote(msg)
    print(f"Execute {msg}")
    generator = generate_iter(1000, msg)

    if not 'uuid' in session:
        session_uuid = str(uuid.uuid4())
        session['uuid'] = session_uuid
        print("generated uuid in ajax request:", session_uuid)
    session_uuid = session['uuid']
    active_generators[session_uuid] = generator

    print("added generator for uuid:", session_uuid)

    return '', 204

@app.route('/')
def index():
    session_uuid = str(uuid.uuid4())
    print("generated uuid in index.html:", session_uuid)
    session['uuid'] = session_uuid

    title="Harry-GPT"
    # model_code = model.code.replace("\n", "<br>  ")

    model_text = ""
    for name, model in models.items():
        model_text += f"<p> <h3>{name}</h5>"
        model_text += "<ul>"
        for layer in model["layers"]:
            model_text += f"<li>{str(layer)}</li>"
        model_text += "</ul></p>"

    model_code = model_text
    return render_template('index.html.j2', title=title,model_code=model_code)

app.run(debug=True)