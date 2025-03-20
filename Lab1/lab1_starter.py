from flask import Flask
from flask import request
from flask import Response
from flask import jsonify
import pandas as pd

from base_iris_lab1 import add_dataset, build, train, score, upload_training, create_and_train_model, retrain_model

app = Flask(__name__)

@app.route('/iris/datasets', methods=['POST'])
def upload_training_route():
    return upload_training(request)


@app.route('/iris/model', methods=['POST'])
def create_and_train_model_route():
    return create_and_train_model(request)

@app.route('/iris/model/<int:model_ID>', methods=['PUT'])
def retrain_model_route(model_ID):
    return retrain_model(model_ID, request)

@app.route('/iris/model/<int:model_ID>/score', methods=['GET'])
def score_model_route(model_ID):
    try:
        fields = request.args.get('fields', '')

        if not fields:
            return jsonify({'error': 'Missing fields parameter'}), 400
        
        field_values = list(map(float, fields.split(',')))

        if len(field_values) != 20:
            return jsonify({'error': 'Exactly 20 fields are required'}), 400

        # score function call
        res = score(model_ID, field_values)

        return res, 200

    except ValueError:
        return jsonify({'error': 'Input is Invalid: Ensure fields are numbers and formatted'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500



if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=4000)


