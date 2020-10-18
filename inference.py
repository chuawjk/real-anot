import ktrain

from datapipeline.preprocessing import Datapipeline


class Model:
    def __init__(self):
        # Load pre-trained model
        print("***Loading preprocessing pipeline")
        self.dpl = Datapipeline()
        print("***Preprocessing pipeline loaded")

        print("***Loading  model***")
        weights_path = "trained_predictor"
        self.model = ktrain.load_predictor(weights_path).model
        self.preproc = ktrain.load_predictor(weights_path).preproc
        self.predictor = ktrain.get_predictor(self.model, self.preproc)
        print("***Model loaded***")

    def get_prediction(self, input_string):
        """ Preprocesses input string, and pass it into model to return real and 
        fake probabilties.

        PARAMS:
            input_string: text string to be passed to model for prediction
        RETURNS:
            output: dict with keys "input_string", and "pred_probs" containing real
                and fake probabilities
        """
        preprocessed_string = self.dpl.preprocess_input(input_string)
        pred_probs = self.predictor.predict_proba(preprocessed_string)[0]
        output = {"input_string": input_string,
                  "pred_probs": pred_probs}
        return output


if __name__ == '__main__':
    text = 'Regularly and thoroughly clean your hands with an alcohol-based hand rub or wash them with soap and water. Why?\
                Washing your hands with soap and water or using alcohol-based hand rub kills viruses that may be on your hands.'
    m = Model()
    output = m.get_prediction(input_string=text)
    print(output["input_string"])
    txt = "The above statement is predicted to be real with {} probability, and fake with {} probability."
    print(txt.format(output["pred_probs"][1], output["pred_probs"][0]))
