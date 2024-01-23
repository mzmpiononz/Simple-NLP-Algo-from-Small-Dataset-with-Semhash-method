import pickle
from flask import Flask, request, jsonify
from utils import semhash_training, inference_preprocess

# the intent name in ENG and DEU
intent_names = ['booking', 'buchung', 'infoline-eng', 'infoline-deu', 'remarks', 'bemerkungen']

# path to the model
path_model = 'eng-deu-model.pkl'
eng_deu_model = pickle.load(open(path_model, 'rb'))

app = Flask(__name__)
@app.route('/', methods=['POST'])
def intent_pred():
    data = request.get_json()
    pred = intent_names[eng_deu_model.predict(inference_preprocess(data['phrase']))[0]]
    return jsonify(pred)
        
if __name__ == '__main__':
    app.run(debug=True, port='5000')