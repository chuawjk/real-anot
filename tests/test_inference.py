import ktrain
import numpy as np
import pytest

from inference import Model


X = ["Regularly and thoroughly clean your hands with an alcohol-based\
        hand rub or wash them with soap and water. Why? Washing your hands\
        with soap and water or using alcohol-based hand rub kills viruses\
        that may be on your hands."]
y = ["real"]
target_names = ["real", "fake"]
t = ktrain.text.Transformer(
    "distilbert-base-uncased", maxlen=107, class_names=target_names)
_ = t.preprocess_train(X, y)
untuned_model = t.get_classifier()


def test_load_weights():
    untuned_weights = np.array(untuned_model.weights[20][0])

    finetuned_model = Model().model
    finetuned_weights = np.array(finetuned_model.weights[20][0])

    weights_changed = (finetuned_weights != untuned_weights).all()

    assert weights_changed, "Model weights same as default; has model been fine-tuned?"


def test_get_prediction():
    model = Model()
    output = model.get_prediction(X)
    output_type = type(output)
    expected_type = dict

    output_keys = list(output.keys())
    expected_keys = ["input_string", "pred_probs"]

    assert (
        output_type == expected_type
    ), f"Expected {expected_type}, returned {output_type} instead"

    assert (
        output_keys == expected_keys
    ), f"Incorrect output dict keys; expected {expected_keys},\
        returned {output_keys} instead"

    assert output["input_string"] == X, f"Output failed to return input string"

    assert (
        np.sum(output["pred_probs"]) == pytest.approx(1)
    ), f"Output probabilities do not sum to 1"