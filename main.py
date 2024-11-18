import tensorflow as tf
import numpy as np
import shap

from gan import get_trained_gan
from model import get_trained_model, NOISE_DIM
from utils import get_train_dataset, get_test_dataset

np.random.seed(42)
tf.random.set_seed(42)

if __name__ == "__main__":

    def explain_classifier_model():
        x, y = get_train_dataset()
        x_test, y = get_test_dataset()

        model = get_trained_model()

        subset = x_test[:10]

        explainer = shap.Explainer(model, x[:100])
        shap_values = explainer(subset)
        print(shap_values)

        for shap_value in shap_values:
            shap.plots.waterfall(shap_value)


    def make_prediction():
        x_test, y_test = get_test_dataset()
        model = get_trained_model()
        print(model.evaluate(x_test, y_test, return_dict=True))


    # Download the data and place in the same folder as "data.zip" and uncomment the following line
    # create_csv()
    # explain_classifier_model()

    generator, predictor = get_trained_gan()

    flow = generator(tf.random.normal([1, NOISE_DIM]), training=False)

    predicted = predictor.predict(flow)
    print(f"Flow {flow}, prediction {predicted}")
