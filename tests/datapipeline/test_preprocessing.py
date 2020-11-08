import numpy as np
import pytest
from sklearn.preprocessing import FunctionTransformer

from datapipeline.preprocessing import Datapipeline


test_string_invalid = "you must log in to continue .."
test_string_valid = "Regularly and thoroughly clean your hands with an alcohol-based\
                hand rub or wash them with soap and water. Why? Washing your hands\
                with soap and water or using alcohol-based hand rub kills viruses\
                that may be on your hands."

dpl = Datapipeline()


def test_remove_invalid_content():
    output_string = dpl.remove_invalid_content(test_string_invalid)

    assert output_string == "", "Artifactual input strings were not sucessfully removed"


def test_pipelinize():
    transformed_func = dpl.pipelinize(dpl.remove_invalid_content)
    transformed_func_type = type(transformed_func)
    expected_type = type(FunctionTransformer(lambda x: x))

    assert (
        transformed_func_type == expected_type
    ), f"Expected output of type {expected_type},\
    returned {transformed_func_type} instead"


def test_preprocess_input():
    output = dpl.preprocess_input(test_string_valid)
    output_type = type(output)
    expected_type = type(np.array([0]))

    assert (
        output_type == expected_type
    ), f"Expected output of type {expected_type},\
    returned {output_type} instead"