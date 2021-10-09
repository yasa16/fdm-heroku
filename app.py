import numpy as np
from flask import Flask, render_template,request, jsonify
import pickle
from flask_cors import CORS, cross_origin

from numpy.core.numeric import cross#Initialize the flask App
app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))
CORS(app)

#default page of our web-app
@app.route('/')
def home():
    return render_template('index.html')

#To use the predict button in our web-app
@app.route('/predict',methods=['POST'])
def predict():
    #For rendering results on HTML GUI

    print(request.form.values())
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    #output = round(prediction[0] ) 
    return  'Customer Segmentation {}'.format(prediction)
    #render_template('index.html', prediction_text=' Customer Segmentation is :{}'.format(prediction))

if __name__ == "__main__":
    app.run(debug=True)


# from flask import Flask
# #from database.db import initialize_db
# from flask_restful import Api
# from Resources.routes import initialize_routes
# from flask_cors import CORS, cross_origin


# app = Flask(__name__)
# api = Api(app)
# cors = CORS(app)

# app.config['CORS_HEADERS'] = 'Content-Type'

# #initialize_db(app)

# initialize_routes(api)

# app.run()






