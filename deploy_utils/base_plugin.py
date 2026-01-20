from typing import Any

class BasePlugin:
    """Base Class for all plugins
    Plugins are used to extend the functionality of the environment.
    Environment will expose internal data to plugins for computation.
    Plugins are all asynchronous with multiprocessing.
    """
    def __init__(self, plugin_name: str, plugin_type: str, 
                 env_type: str | None = None,
                 **kwargs):
        self.plugin_name = plugin_name
        self.plugin_type = plugin_type

        if env_type is None:
            raise RuntimeError("Environment type is not specified")
        supported_env_types = ['mujoco', 'unitree']
        assert env_type in supported_env_types, f"Environment type {env_type} is not supported. Supported types are: {supported_env_types}"
        self.env_type = env_type

        supported_plugin_types = ['sensor']
        assert plugin_type in supported_plugin_types, f"Plugin type {plugin_type} is not supported. Supported types are: {supported_plugin_types}"
        self.plugin_type = plugin_type

        for key, value in kwargs.items():
            setattr(self, key, value)

    def initialize(self, **kwargs) -> None:
        """Initialize the plugin. Initialization happens in the main process."""
        raise NotImplementedError("This function should be implemented by the subclass")

    def run(self) -> None:
        """Run the plugin. Running happens in the subprocess."""
        raise NotImplementedError("This function should be implemented by the subclass")