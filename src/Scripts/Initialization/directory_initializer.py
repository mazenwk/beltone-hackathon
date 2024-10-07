import os


class DirectoryInitializer:
    """
    A class to initialize directories within a specified path.

    Attributes:
        path (str): The base path where directories will be created.
        dirs (list): A list of directory names to create within the specified path.
    """

    def __init__(self, path: str = None, dirs: list[str] = None):
        """
        Initializes a new instance of the DirectoryInitializer class.

        Args:
            path (str): The base path where directories will be created.
                Defaults to the current working directory if not provided.
            dirs (list[str], optional): A list of directory names to create.
                Defaults to ['Output', 'Pickles'] if not provided.
        """
        if path is None:
            path = os.getcwd()
        if dirs is None:
            dirs = ['Output', 'Pickles']

        self.path = path
        self.dirs = dirs

        self.initialize()

    def initialize(self):
        """
        Initializes the directory creation process.
        """
        print(f'Initializing directories in {self.path}{os.sep}')
        self._create_directories()

    def _create_directories(self):
        """
        Creates the specified directories in the base path.
        """
        for dir_name in self.dirs:
            dir_path = os.path.join(self.path, dir_name)
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
                print(f'Created directory {dir_path}')
            else:
                print(f'Directory {dir_path} already exists')

# Example usage:
# if __name__ == '__main__':
#     directory = DirectoryInitializer('../../')
