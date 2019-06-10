from flask import Flask, request, jsonify
from flask_restful import Resource, Api

app = Flask(__name__)
api = Api(app)

def getData():
    obj = {}
    obj['first_name'] = "John"
    obj['last_name'] = "Smith"
    obj['age'] = 34
    return obj

class Classify(Resource):
    def get(self): # get is the right http verb because it caches the outputs and is faster in general
        data = request.get_json() # reading the data
        print(data)
        result = getData() # conversion to list because numpy.ndarray cannot be jsonified
        if (data['param1']):
            result['param'] = data['param1']
        if (data['param2']):
            i = int(data['param2'])
            num = 2
            for i in range(i):
                num = num * 2
            result['val'] = num
        return jsonify(result) # returning the result of classification

api.add_resource(Classify, '/classify')

app.run(port=5001)