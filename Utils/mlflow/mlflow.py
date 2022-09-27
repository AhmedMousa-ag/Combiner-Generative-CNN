import mlflow


# Will be used as decorator
def mlflow_track(func):
    """This function is a decorator to any experiment we want to track"""

    def track_exp(*args, **kwargs):
        with mlflow.start_run():
            func(*args, **kwargs)

    return track_exp
