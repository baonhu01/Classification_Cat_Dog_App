from flask import Flask, render_template, request, send_from_directory
import cv2
from keras.models import load_model
import numpy as np


model = load_model('static/model.h5')

COUNT = 0
app = Flask(__name__)
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 1

@app.route('/')
def man():
    return render_template('index.html')


@app.route('/home', methods=['POST'])
def home():
    global COUNT
    img = request.files['image']

    img.save('static/{}.jpg'.format(COUNT))    
    img_arr = cv2.imread('static/{}.jpg'.format(COUNT))

    img_arr = cv2.resize(img_arr, (128,128))
    img_arr = img_arr / 255.0
    img_arr = img_arr.reshape(1, 128,128,3)
    prediction = model.predict(img_arr)

    print(prediction)

    x = round(prediction[0,0], 2) #lam tron 2 chu so
    y = round(prediction[0,1], 2)
    preds = np.array([x,y])

    print(preds)

    COUNT += 1
    return render_template('prediction.html', data=preds)


@app.route('/load_img')
def load_img():
    global COUNT
    return send_from_directory('static', "{}.jpg".format(COUNT-1))


if __name__ == '__main__':
    app.run(debug=True)



