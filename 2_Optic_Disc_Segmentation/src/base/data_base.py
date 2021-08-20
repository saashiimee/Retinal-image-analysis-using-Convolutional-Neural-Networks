class DataLoaderBase(object):

    def __init__(self, config):
        self.config = config

    def prepare_dataset(self):
        raise NotImplementedError

    def get_train_data(self):
        raise NotImplementedError

    def get_validate_data(self):
        raise NotImplementedError
