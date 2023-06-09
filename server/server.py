#!/usr/bin/env python3
from flask import Flask, render_template, request, Response, session, copy_current_request_context
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
    }
    models[model_name] = {"model": model, "layers": layers}

# model = list(models.values())[0]["model"]
model = models["lstm-2-256-de"]["model"]

active_generators = {}

def generate_it(model: nn.Module, length: int, prompt: str = "") -> Iterator[str]:
    if prompt == "":
        prompt = model.ix_to_char[torch.randint(low=0, high=model.vocab_size, size=(1,)).item()]
    char, context = model.generate_start(prompt)
    yield char
    for _ in range(length - 1):
        char, context = model.generate_step(context)
        yield char

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
    generator = generate_it(model=model, length=1000, prompt=msg)

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

    # model_code = model.code.replace("\n", "<br>  ")

    # model_text = ""
    # for name, model in models.items():
    #     model_text += f"<p> <h3>{name}</h5>"
    #     model_text += "<ul>"
    #     for layer in model["layers"]:
    #         model_text += f"<li>{str(layer)}</li>"
    #     model_text += "</ul></p>"
    # model_code = model_text

    model_data = {}
    for value in models_json_data.values():
        model_data[value["name"]] = value["loss"]

    return render_template('index.html.j2', models_json=json.dumps(models_json_data), model_data=model_data)

app.run(debug=True)