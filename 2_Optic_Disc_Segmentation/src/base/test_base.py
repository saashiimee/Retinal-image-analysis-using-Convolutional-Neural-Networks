class TestBase(object):

    def __init__(self, config):
        self.config = config

    def load_model(self):
        raise NotImplementedError

    def analyze_name(self, path):
        raise NotImplementedError

    def predict(self):
        raise NotImplementedError
