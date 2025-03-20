from flask import Flask
from flask import request
from flask import Response
from flask import abort
from flask import jsonify

app = Flask(__name__)

employee_names = []
employee_nicknames = []

# the minimal Flask application
@app.route('/')
def index():
    return Response( '<h1>Hello, World!</h1>', status=201 )


# bind multiple URL for one view function
@app.route('/hi')
@app.route('/hello')
def say_hello():
    return Response('<h1>Hello, Flask!</h1>', status=201)


# dynamic route, URL variable default
@app.route('/greet', defaults={'name': 'Programmer'})
@app.route('/greet/<name>')
def greet(name):
    ret = 'Hello ' + name
    print(ret)
    return Response(ret, status=201)

@app.route('/employee', methods=['POST'])
def post_example():
    if 'name' in request.form:
        emp_name = request.form['name']
    else:
        emp_name = "John Doe"
    if 'nickname' in request.form:
        emp_nickname = request.form['nickname']
    else:
        emp_nickname = "None"

    print('The employee name is ' + emp_name + ' and their nickname is ' + emp_nickname )

    employee_names.append(emp_name)
    employee_nicknames.append(emp_nickname)

    ret_val = { 'name' : emp_name, 'nickname' : emp_nickname, 'employee_ID': len(employee_names) - 1 }
    return jsonify (ret_val)

# dynamic route, URL variable default
@app.route('/employee/<ID>')
def employee_greet(ID):
    greet_type = request.args.get('greet_type')
    if greet_type == None:
        greet_type = 'formal'

    name = employee_names[int(ID)]
    nickname = employee_nicknames[int(ID)]

    if greet_type == 'formal':
        ret = 'Hello ' + name
    else:
        ret = 'Hello ' + nickname + ' !'

    print(ret)
    return Response(ret, status=201)

@app.route('/employee/<emp_ID_str>', methods=['PUT'])
def put_example(emp_ID_str):
    emp_ID = int(emp_ID_str)

    if emp_ID >= len(employee_names):
        return Response("Invalid employee ID", status=401)
    emp_name = employee_names[emp_ID]
    emp_nickname = employee_nicknames[emp_ID]
    
    if 'name' in request.form:
        emp_name = request.form['name']
        employee_names[emp_ID] = emp_name

    if 'nickname' in request.form:
        emp_nickname = request.form['nickname']
        employee_nicknames[emp_ID] = emp_nickname

    ret_val = { 'name' : emp_name, 'nickname' : emp_nickname, 'employee_ID': emp_ID }
    return jsonify (ret_val)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=4000)
