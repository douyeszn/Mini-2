import tensorflow as tf
from tensorflow.keras import layers
import pandas as pd
import numpy as np
from tensorflow.keras import datasets, layers
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from flask import jsonify
import io
from datetime import datetime
import boto3
from botocore.config import Config

print('starting up iris model service')

my_config = Config(
    region_name = 'eu-central-1'
)

# Get the service resource.
session = boto3.Session(
    aws_access_key_id='<Replace-key>',
    aws_secret_access_key='<Replace-key>',
)

dynamodb = session.resource('dynamodb', config=my_config)
scores_table = dynamodb.Table('IrisExtended')

global models, datasets
models = []
datasets = []

def upload_training(request):
    global datasets
    
    if 'train' not in request.form:
        return jsonify({'error': 'No file uploaded under form field "train"'}), 400

    train_file = request.form.get('train')

    try:
        df = pd.read_csv(train_file)

        dataset_id = len(datasets)
        datasets.append(df)

        return jsonify({'dataset_id': dataset_id}), 201 
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def create_and_train_model(request):
    try:
        dataset_ID = int(request.form.get('dataset', -1))
        print('dataset id from request',dataset_ID, request.form.get('dataset'))
        if dataset_ID < 0 or dataset_ID >= len(datasets):
            return jsonify({'error': 'Invalid dataset index'}), 400

        model_ID = build()
        print('id',model_ID)
        train(model_ID, dataset_ID)

        return jsonify({'model_id': model_ID}), 201
    except Exception as e:
        return jsonify({'error': str(e)}), 500
        pass

def retrain_model(model_ID, request):
    print('Retraining model:', model_ID)
    try:
        dataset_ID = int(request.args.get('dataset', -1))

        if model_ID < 0 or model_ID >= len(models):
            return jsonify({'error': 'Invalid model index'}), 400
        if dataset_ID < 0 or dataset_ID >= len(datasets):
            return jsonify({'error': 'Invalid dataset index'}), 400

        history = train(model_ID, dataset_ID) 

        return jsonify({'training_history': history}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


def build():
    global models

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(10, activation='relu', input_shape=(20,)),
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(3, activation='softmax')
        ])

    model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

    models.append( model )
    model_ID = len(models) - 1

    return model_ID

def load_local():
    global datasets

    print("load local default data")

    dataFolder = './'
    dataFile = dataFolder + "iris.data"

    datasets.append( pd.read_csv(dataFile) )
    return len( datasets ) - 1

def add_dataset( df ):
    global datasets

    datasets.append( df )
    return len( datasets ) - 1

def get_dataset( dataset_ID ):
    global datasets

    return datasets[dataset_ID]

def train(model_ID, dataset_ID):
    global datasets, models
    dataset = datasets[dataset_ID]
    model = models[model_ID]

    # return
    X = dataset.iloc[:,1:].values
    y = dataset.iloc[:,0].values

    encoder =  LabelEncoder()
    y1 = encoder.fit_transform(y)
    Y = pd.get_dummies(y1).values

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

    history = model.fit(X_train, y_train, batch_size=1, epochs=10)
    print(history.history)

    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print('Test loss:', loss)
    print('Test accuracy:', accuracy)

    y_pred = model.predict(X_test)

    actual = np.argmax(y_test,axis=1)
    predicted = np.argmax(y_pred,axis=1)
    print(f"Actual: {actual}")
    print(f"Predicted: {predicted}")

    conf_matrix = confusion_matrix(actual, predicted)
    print('Confusion matrix on test data is {}'.format(conf_matrix))
    print('Precision Score on test data is {}'.format(precision_score(actual, predicted, average=None)))
    print('Recall Score on test data is {}'.format(recall_score(actual, predicted, average=None)))

    return(history.history)

def score( model_ID, r ):
    global models
    model = models[model_ID]

    x_test2 = np.array( [r] )

    y_pred2 = model.predict(x_test2)
    print(y_pred2)
    iris_class = np.argmax(y_pred2, axis=1)[0]
    print(iris_class)

    return "Score done, class=" + str(iris_class)

def new_model(dataset):
    model_id = build()
    train(model_id, dataset)
    return model_id

metrics = []

def save_metrics( model_id, metrics_bundle ):
    if model_id < len(metrics):
        metrics[model_id] = metrics_bundle
    else:
        metrics.append(metrics_bundle)
    return

def post_score(log_table, feature_string, class_string, actual_string, prob_string ):
    current_time = datetime.utcnow().strftime('%H:%M:%S.%f')[:-3]

    response = log_table.put_item(
       Item={
            'partition_key': current_time,
            'sort_key': "abc",
            'Features': feature_string,
            'Class' : class_string,
            'Actual' : actual_string,
            'Probability' : prob_string
            }
    )
    print("PutItem response:", response)
    return response

def test(model_id, dataset_id):
    """
    Test a model with a dataset and log results to DynamoDB.
    Maintains original API signature for compatibility with existing endpoints.
    """
    global models, datasets, metrics

    model = models[int(model_id)]
    df = datasets[int(dataset_id)]
    
    X_test = df.iloc[:,1:].values
    y = df.iloc[:,0].values

    encoder = LabelEncoder()
    y1 = encoder.fit_transform(y)
    y_test = pd.get_dummies(y1).values

    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print('Test loss:', loss)
    print('Test accuracy:', accuracy)

    y_pred = model.predict(X_test)

    actual = np.argmax(y_test,axis=1)
    predicted = np.argmax(y_pred,axis=1)
    print(f"Actual: {actual}")
    print(f"Predicted: {predicted}")

    # Save overall metrics
    save_metrics(model_id, {'accuracy': float(accuracy), 'actual': actual.tolist(), 'predicted': predicted.tolist()})
    
    # Post individual records to DynamoDB
    try:
        # Try to import necessary components for DynamoDB logging
        try:         
            # Process each record and post to DynamoDB
            for i in range(len(df)):
                # Create feature dictionary from all feature columns
                feature_dict = {}
                for col_idx, value in enumerate(df.iloc[i, 1:]):
                    # Use column name if available, otherwise use generic feature name
                    if len(df.columns) > col_idx + 1:
                        col_name = df.columns[col_idx + 1]
                    else:
                        col_name = f"feature_{col_idx}"
                    feature_dict[col_name] = value
                
                # Convert to string format as required
                feature_string = str(feature_dict)
                
                # Get predicted class (convert numeric prediction back to class name)
                predicted_index = predicted[i]
                predicted_class = encoder.inverse_transform([predicted_index])[0]
                class_string = str(predicted_class)
                
                # Get actual class
                actual_string = str(y[i])
                
                # Get probability for the predicted class
                prob_value = y_pred[i][predicted_index]
                prob_string = str(prob_value)
                
                # Call post_score function to write to DynamoDB
                post_score(scores_table, feature_string, class_string, actual_string, prob_string)
                
                print(f"Posted record {i+1}/{len(df)} to DynamoDB")
        
        except (ImportError, AttributeError) as e:
            print(f"Could not import required components for DynamoDB logging: {str(e)}")
            print("Individual records will not be posted to DynamoDB.")
            
    except Exception as e:
        print(f"Error in DynamoDB logging: {str(e)}")
        print("Continuing with test function...")

    # Return the metrics as before to maintain API compatibility
    return metrics[model_id]