#!/usr/bin/env python3
from flask import Flask, render_template, request, Response, session, copy_current_request_context, send_from_directory
import urllib.parse
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import time
import secrets
import uuid
import os
from typing import List, Tuple, Iterator

app = Flask(__name__)
app.secret_key = secrets.token_hex(16)

models = {} # key: name, value: {model, [{"type", **params}]}
models_json_data = {} # data to send to the client

for model_filename in os.listdir("models"):
    if not model_filename.endswith(".pt"):
        continue
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

    models_json_data[model_name] = {
        "layers": layers,
        "name": model.info_name,
        "epochs": model.info_epochs,
        "time": model.info_time,
        "loss": model.info_loss,
        "batch_size": model.info_batch_size,
        "is_snapshot": model.info_is_snapshot,
        "img_url": f"/img/{model_name}.png"
    }
    models[model_name] = {"model": model, "layers": layers}

default_model_name = "gru-res-6-256-de"
model = models[default_model_name]["model"]

active_generators = {}
active_text = {}

def generate_it(model: nn.Module, length: int, prompt: str = "") -> Iterator[str]:
    if prompt == "":
        prompt = model.ix_to_char[torch.randint(low=0, high=model.vocab_size, size=(1,)).item()]
    
    # TODO: Convert to ascii. We should find a better way to handle unicode chars
    prompt = prompt.encode("ascii", "replace").decode("utf-8", "replace")
    prompt = prompt.replace("\ufffd", "")

    char, context = model.generate_start(prompt)
    yield char
    for _ in range(length - 1):
        char, context = model.generate_step(context)
        yield char

@app.route('/stream')
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
        while True:
            try:
                if uuid in active_generators:
                    text = next(active_generators[uuid])
                    active_text[uuid] += text
                    yield f"event: gen\ndata: <span>{text}</span>\n\n"
                else:
                    yield "event: keeapalive\ndata: \n\n"
                    time.sleep(0.1)
            except StopIteration:
                yield "event: end of iterator\ndata: \n\n"
                active_generators.pop(uuid)
                time.sleep(0.1)
    return Response(outer_gen(session_uuid), mimetype='text/event-stream')

def create_chat_generator(prompt: str) -> Iterator[str]:
    print("Execute", prompt)
    generator = generate_it(model=model, length=1000, prompt=prompt)

    if not 'uuid' in session:
        session_uuid = str(uuid.uuid4())
        session['uuid'] = session_uuid
        print("generated uuid in ajax request:", session_uuid)
    session_uuid = session['uuid']
    active_text[session_uuid] = ""
    active_generators[session_uuid] = generator
    print("added generator for uuid:", session_uuid)

@app.route('/chat', methods = ['POST'])
def chat_receive_message():
    msg = request.form["msg"]
    msg = urllib.parse.unquote(msg)
    create_chat_generator(msg)
    return '', 204 # No content

@app.route('/chat/continue', methods = ['POST'])
def chat_continue():
    create_chat_generator(active_text[session['uuid']])
    return '', 204 # No content

@app.route("/img/<filename>.png")
def img_static(filename):
    return send_from_directory(f"{app.root_path}/models/", f"{filename}.png")

@app.route('/')
def index():
    session_uuid = str(uuid.uuid4())
    session['uuid'] = session_uuid

    model_data = {}
    for value in models_json_data.values():
        model_data[value["name"]] = value["loss"]

    return render_template('index.html.j2',
        models_json=json.dumps(models_json_data),
        model_data=model_data,
        default_model_name=default_model_name)

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)