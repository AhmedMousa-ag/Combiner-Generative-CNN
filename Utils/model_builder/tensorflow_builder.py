from model_builder import model_builder


class tf_builder(model_builder):
    def __init__(self, *args, **kwargs):
        super.__init__(*args,**kwargs)

    def _compile(self):
        self.model.compile(
            loss=self.loss,
            optimizer=self.optimizer,
            metrics=self.metrics)
