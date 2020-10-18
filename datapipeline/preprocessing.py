import re

import numpy as np
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline

class Datapipeline():
    def __init__(self):
        pass

    def remove_invalid_content(self, text):
        """Removes artefacts from input string.

        PARAMS:
            text: input string

        RETURNS:
            output: original string if text does not contain artefacts; empty
                string if text contains artefactual content
        """
        invalid_content = ["we 've detected that javascript is disabled in your browser. would you like to proceed to legacy twitter.",
                           "do you want to join facebook ?.",
                           "you must log in to continue ..",
                           "join this group to post and comment ..",
                           "this website is using a security service to protect itself from online attacks ..",
                           "see more.+on facebook"]

        for phrase in invalid_content:
            phrase = re.compile(phrase)
            phrase_match = bool(re.match(phrase, str(text)))

            if phrase != phrase:  # test for NaN
                output = ""
                break
            elif phrase_match == True:
                output = ""
                break
            else:
                output = text

        return output


    class list_compre_wrapper(object):
        def __init__(self, function):
            self.function = function

        def __call__(self, list_or_series):
            return [self.function(i) for i in list_or_series]


    def pipelinize(self, function):
        """Wraps function for integration with sklearn's Pipeline()

        PARAMS:
            function: function to be wrapped
        RETURNS:
            transformed_func: function wrapped with FunctionTransformer()
        """
        lcw = self.list_compre_wrapper(function)
        transformed_func = FunctionTransformer(lcw)
        return transformed_func


    def preprocess_input(self, input_string):
        """Preprocesses input data in preparation for for inference.
        
        PARAMS:
            input_string: str containing the text to be classified
        RETURNS:
            X: numpy dense array
        """
        X = np.array(input_string).reshape([1, -1])
        remove_inval = Pipeline(
            [('remove_invalid_content', self.pipelinize(self.remove_invalid_content))])
        X = remove_inval.fit_transform(X)[0]

        return X
