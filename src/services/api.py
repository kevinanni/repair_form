from flask import Flask, jsonify, request
from src.business_logic.model_management import load_model, make_prediction

app = Flask(__name__)


@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.json  # 假设数据是以JSON格式传递的
    model = load_model()
    predictions = make_prediction(data, model)
    return jsonify({'predictions': predictions.tolist()})


if __name__ == '__main__':
    app.run(debug=True)
