from sklearn.metrics import accuracy_score
from LSP_model import predict


def test_predict(sample_input_data):
    test_inputs = sample_input_data
    prediction = predict.make_prediction(test_data=test_inputs)
    accuracy = accuracy_score(test_inputs['LANDSLIDE'], prediction)
    assert accuracy > 0.90
