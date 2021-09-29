#!/usr/bin/env python3

from flask import Flask, jsonify, render_template, Response
from flask_restful import Resource, Api, marshal_with
from flask_socketio import SocketIO, emit, send
from flask_cors import CORS
from flask import request
from threading import Thread
import time
from calculations import *
import json


app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, cors_allowed_origins='*')
thread = None
clients = 0
CORS(app)

def ini_socket():
    global clients, thread
    thread = None


@app.route('/api/socket')
def index():
    print('Route socket init')
    global thread
    if thread is None:
        thread = Thread(target=ini_socket)
        thread.start()
    return ('{"ok":"success"}')


# Function that runs when a clients get connected to the server
@socketio.on('connect')
def test_connect():
    global clients
    clients += 1
    print('Client connected test')


# Read data from client
@socketio.on('new-message')
def handle_message(message):
    print('received message' + message)
    send_data()


# Send data to client
@socketio.on('new-message-s')
def send_data():
    emit('data-msg', {'message': 'message'})


@socketio.on('disconnect')
def test_disconnect():
    global clients
    clients -= 1
    print('Client disconnected')

@app.route('/table1', methods=['POST'])
def update_risks_table():
    print(json.loads(request.data))
    new_json = request.data
    try:
        if BaseClass.risks_table.read_from_json(new_json, table_name='risks'):
            BaseClass.is_relevant = False
            return '{"text":"All good"}', 200
        else:
            return '{"text":"Error!!"}', 500
    except:
        return '{"text":"I am fallen, I can\'t get up!!"}', 500

@app.route('/table2', methods=['POST'])
def update_costs_table():
    print(json.loads(request.data))
    new_json = request.data
    try:
        if BaseClass.costs_table.read_from_json(new_json, table_name='costs'):
            BaseClass.is_relevant = False
            return '{"text":"All good"}', 200
        else:
            return '{"text":"Error!!"}', 500
    except:
        return '{"text":"I am fallen, I can\'t get up!!"}', 500

@app.route('/table3', methods=['POST'])
def update_reasoning_table():
    print(json.loads(request.data))
    new_json = request.data
    try:
        if BaseClass.reasons_table.read_from_json(new_json, table_name='reasoning'):
            return '{"text":"All good"}', 200
        else:
            return '{"text":"Error!!"}', 500
    except:
        return '{"text":"I am fallen, I can\'t get up!!"}', 500

@app.route('/table1', methods=['GET'])
def send_table_with_risks():
    response = jsonify(BaseClass.risks_table.convert_numpy_to_json())
    response.headers.add("Access-Control-Allow-Origin", "*")
    return response

@app.route('/table2', methods=['GET'])
def send_table_with_costs():
    response = jsonify(BaseClass.costs_table.convert_numpy_to_json())
    response.headers.add("Access-Control-Allow-Origin", "*")
    return response

@app.route('/table3', methods=['GET'])
def send_table_with_reasons():
    response = jsonify(BaseClass.reasons_table.convert_numpy_to_json())
    response.headers.add("Access-Control-Allow-Origin", "*")
    return response

@app.route('/plot')
def plot_png():
    if not BaseClass.is_relevant:
        BaseClass.optimize_for_all_costs(multiprocessing_mode=True)
    BaseClass.save_optimal_strategy_curve()
    return {'text':"success"}, 200

@app.route('/table4', methods=['GET'])
def send_table_with_optimization():
    target_budget = float(json.loads(request.data)["number"])
    strategy_table, risk, cost = BaseClass.get_optimal_strategy_with_risk_and_cost_given_budget(target_budget)
    all_together = dict()  # json.loads(strategy_table)
    all_together['table'] = strategy_table
    all_together["risk"] = risk
    all_together["cost"] = cost
    all_together = json.dumps(all_together)
    response = jsonify(all_together)
    return response

# This is the function that will create the Server in the ip host and port 5000
if __name__ == "__main__":
    print("starting webservice")
    app.run(debug=True, port="12345", host="0.0.0.0")
