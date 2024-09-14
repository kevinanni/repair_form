from sklearn.linear_model import LinearRegression
import joblib


def load_model():
    """Load the trained model."""
    try:
        model = joblib.load('path/to/model.pkl')
    except FileNotFoundError:
        model = LinearRegression()  # 示例模型
        joblib.dump(model, 'path/to/model.pkl')
    return model


def make_prediction(data, model):
    """Make predictions using the model."""
    predictions = model.predict(data)
    return predictions
