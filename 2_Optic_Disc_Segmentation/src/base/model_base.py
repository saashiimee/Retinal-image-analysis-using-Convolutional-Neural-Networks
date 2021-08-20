class ModelBase(object):

    def __init__(self, config):
        self.config = config
        self.model = None

    def save(self):
        if self.model is None:
            raise Exception("[Error] No Model Found. Please build the model first before proceeding.")

        print("[INFO] Saving Model...")
        json_string = self.model.to_json()
        open(self.config.checkpoint + self.config.model_name + "_" + self.config.dataset_name + '_architecture.json', 'w').write(json_string)
        print("[INFO] Model Saved. Your model can be found in the 'models' folder.")

    def load(self):
        if self.model is None:
            raise Exception("[Error] No Model Found. Please build the model first before proceeding.")

        print("[INFO] Loading Model Checkpoint ...\n")
        self.model.load_weights(self.config.hdf5_path + self.config.model_name + "_" + self.config.dataset_name + '_best_weights.h5')
        print("[INFO] Model Loaded. Ready to proceed.")

    def build_model(
            self,
            input_shape,
            num_classes,
            activation,
            use_batch_norm,
            upsample_mode,
            dropout,
            dropout_change_per_layer,
            dropout_type,
            use_dropout_on_upsampling,
            use_attention,
            filters,
            num_layers,
            output_activation):
        raise NotImplementedError
