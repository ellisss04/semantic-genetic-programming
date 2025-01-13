import yaml


class Config:
    def __init__(self, config_path):
        """
        Initialize the Config object by loading parameters from the YAML file.

        Args:
            config_path (str): Path to the YAML configuration file.
        """
        self.config_path = config_path
        self.config = self.load_config()

    def load_config(self):
        """
        Load the YAML configuration file.

        Returns:
            dict: Loaded configuration parameters.
        """
        try:
            with open(self.config_path, 'r') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            raise FileNotFoundError(f"Config file not found at: {self.config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML file: {e}")

    def get(self, key, default=None):
        """
        Retrieve a configuration value.

        Args:
            key (str): The key to look up in the configuration.
            default (Any): Default value to return if the key is not found.

        Returns:
            Any: The value associated with the key or the default value.
        """
        return self.config.get(key, default)

    def update(self, updates: dict):
        """
        Update the configuration in memory and save changes to the YAML file.

        Args:
            updates (dict): Dictionary of parameters to update.
        """
        self.config.update(updates)
        self.save_config()

    def save_config(self):
        """
        Save the current configuration back to the YAML file.
        """
        with open(self.config_path, 'w') as file:
            yaml.safe_dump(self.config, file)
