from flask import Flask, render_template, request
import pickle
import numpy as np

model = pickle.load(open('model.pkl', 'rb'))

app = Flask(__name__)



@app.route('/')
def man():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def home():
    data1 = request.form['Variable 1']
    data2 = request.form['Variable 2']
    data3 = request.form['Variable 3']
    data4 = request.form['Variable 4']
    data5 = request.form['Variable 5']
    data6 = request.form['Variable 6']
    arr = np.array([[data1, data2, data3, data4, data4, data6]])
    pred = model.predict(arr)
    return render_template('gambas.html', variable=pred)


if __name__ == "__main__":
    app.run(debug=True)