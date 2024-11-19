import tensorflow as tf

# Custom layer for supporting proper saving of the SciBERT's layers within the model too.
@tf.keras.utils.register_keras_serializable()
class ExtractCLSOutput(tf.keras.layers.Layer):
    def __init__(self, scibert_model, hidden_size=768, **kwargs):
        super(ExtractCLSOutput, self).__init__(**kwargs)
        self.scibert_model = scibert_model
        self.hidden_size = hidden_size

    def call(self, inputs, **kwargs):
        input_ids, attention_mask = inputs
        outputs = self.scibert_model(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]  # Extract [CLS] token output
        return cls_output

    def get_config(self):
        # Serialize necessary configuration
        config = super(ExtractCLSOutput, self).get_config()
        config.update({
            "hidden_size": self.hidden_size,
            "scibert_model": tf.keras.utils.serialize_keras_object(self.scibert_model)
        })
        return config

    @classmethod
    def from_config(cls, config):
        # Deserialize the scibert_model
        scibert_model = tf.keras.utils.deserialize_keras_object(config.pop("scibert_model"))
        return cls(scibert_model=scibert_model, **config)

# Custom objects for loading the model
model_custom_objects = {
    'ExtractCLSOutput':   ExtractCLSOutput
}

class ModelLoader:
    _model = None

    @staticmethod
    def get_model():

        MODEL_PATH = "/home/admin/databases/task_api_virtual_environment/api_demo/data/model08.keras"

        if ModelLoader._model is None:
            #
            ##ModelLoader._model = tf.keras.models.load_model( MODEL_PATH  )
            ModelLoader._model = tf.keras.models.load_model( MODEL_PATH, custom_objects = model_custom_objects )

        return ModelLoader._model
