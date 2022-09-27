class model_builder():
    def __init__(self, loss_func, optimizer, metrics, pic_shape=[344, 344]):
        self.loss = loss_func
        self.optimizer = optimizer
        self.metrics = metrics
        self.pic_shape = pic_shape
        self.model = None

    def _build_model(self):
        pass

    def _compile(self):
        pass

    def get_model(self):
        return self.model
