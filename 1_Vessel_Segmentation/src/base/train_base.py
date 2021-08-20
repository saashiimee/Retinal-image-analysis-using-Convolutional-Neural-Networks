class TrainBase(object):

    def __init__(self, config, model, data):
        self.config = config
        self.model = model
        self.data = data

    def train(self):
        raise NotImplementedError
