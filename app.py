from flask import Flask, request,  render_template
from waitress import serve
from inference import Model
import numpy as np

app = Flask(__name__)
model = Model()


@app.route('/', methods=['GET', 'POST'])
def index():
    """Renders the app homepage.

    PARAMS:
        None
    RETURNS: 
        render_template: if POST - Rendering of homepage with results
                         else GET - Rendering of homepage without results
    """
    if request.method == 'POST':
        output = predict()
        orig_text = output["input_string"]
        real_percent = str(round(output["pred_probs"][1] * 100))
        fake_percent = str(round(output["pred_probs"][0] * 100))
        result_text = real_percent + "% real, " + fake_percent + "% fake"
        return render_template('index_post.html',
                               orig_text=orig_text,
                               result_text=result_text)

    return render_template('index_get.html')


@app.route('/predict', methods=['POST'])
def predict():
    """Calls inference for prediction
    
    PARAMS:
        None
    RETURNS:
        output: dict with keys "input_string", and "pred_probs" containing real
            and fakeprobabilities
    """
    text = request.form['text']
    if text == "":
        text = 'Regularly and thoroughly clean your hands with an alcohol-based\
                hand rub or wash them with soap and water. Why? Washing your hands\
                with soap and water or using alcohol-based hand rub kills viruses\
                that may be on your hands.'
    output = model.get_prediction(text)
    return output


if __name__ == "__main__":
    serve(app, host="0.0.0.0", port=8000)
