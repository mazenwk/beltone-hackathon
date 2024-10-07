import os


class DirectoryInitializer:
    def __init__(self, path: str, dirs=None):
        if dirs is None:
            dirs = ['Output', 'Pickles']
        self.path = path
        self.dirs = dirs

        self.initialize()

    def initialize(self):
        print(f'Initializing directories in {self.path}{os.sep}')
        self._create_directories()

    def _create_directories(self):
        for dir in self.dirs:
            dir_path = os.path.join(self.path, dir)
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
                print(f'Created directory {dir_path}')
            else:
                print(f'Directory {dir_path} already exists')

# Example usage
# if __name__ == '__main__':
#     directory = DirectoryInitializer('../../')
