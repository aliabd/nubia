from nubia import Nubia
import gradio


def load():
    nubia = Nubia()
    return nubia


def predict(inp_1, inp_2, nubia):
    features = nubia.score(inp_1, inp_2, get_features=True)
    labels = {k:v for k,v in features["features"].items()}
    labels["nubia_score"] = features["nubia_score"]
    labels = {"nubia_score": features["nubia_score"]}
    return labels


INPUTS = [gradio.inputs.Textbox(), gradio.inputs.Textbox()]
OUTPUTS = gradio.outputs.Label()
INTERFACE = gradio.Interface(fn=predict, inputs=INPUTS, outputs=OUTPUTS,
                             load_fn=load, capture_session=True)

INTERFACE.launch(inbrowser=True)
