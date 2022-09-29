import yaml


def load_config_file(path="config.yaml"):
    """This function load our config yaml file
    Args:
        path to our configuration file, it assumes by default that's in the same working directory // Change if it's else where.
    :returns configle file to that looks where
    """
    with open(path, "r") as fw:
        config_file = yaml.safe_load(fw)
    return config_file
    # -----------


def write_to_yaml(write_file, path="config.yaml"):
    """ It writes dynamically our specified configuration to our config file
    Args:
        write_file: the configuration you want to write into our config file
        path: path to our configuration file, it assumes by default that's in the same working directory // Change if it's else where.
    """
    with open(path, "w") as fw:
        yaml.dump(write_file, fw)
