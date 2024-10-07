import os
import logging


class DirectoryInitializer:
    """
    A class to initialize directories within a specified path.

    Attributes:
        path (str): The base path where directories will be created.
        dirs (list[str]): A list of directory names to create within the specified path.
    """

    def __init__(self, path: str = None, dirs: list[str] = None):
        """
        Initializes a new instance of the DirectoryInitializer class.

        Args:
            path (str, optional): The base path where directories will be created.
                Defaults to the current working directory if not provided.
            dirs (list[str], optional): A list of directory names to create.
                Defaults to ['Output', 'Pickles'] if not provided.
        """
        if path is None:
            path = os.getcwd()
        if dirs is None:
            dirs = ['Output', 'Pickles', 'Data']

        self.path = os.path.normpath(os.path.abspath(path))
        self.dirs = dirs

        self._validate_path()

    def _validate_path(self):
        """
        Validates that the base path exists and is a directory.
        Raises an error if the path is invalid.
        """
        if not os.path.exists(self.path):
            raise ValueError(f"The provided path '{self.path}' does not exist.")
        if not os.path.isdir(self.path):
            raise ValueError(f"The provided path '{self.path}' is not a directory.")

    def initialize(self):
        """
        Initializes the directory creation process by creating specified directories.
        """
        logging.info(f'Initializing directories in {self.path}')
        self._create_directories()

    def _create_directories(self):
        """
        Creates the specified directories in the base path.
        """
        for dir_name in self.dirs:
            dir_path = os.path.join(self.path, dir_name)
            try:
                if not os.path.exists(dir_path):
                    os.makedirs(dir_path)
                    logging.info(f'Created directory: {dir_path}')
                else:
                    logging.info(f'Directory already exists: {dir_path}')
            except OSError as e:
                logging.error(f'Failed to create directory {dir_path}: {e}')


# Example usage:
# if __name__ == '__main__':
#     logging.basicConfig(level=logging.INFO)
#     directory = DirectoryInitializer('../../')
#     directory.initialize()
