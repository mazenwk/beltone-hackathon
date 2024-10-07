import os.path

from Scripts.Initialization.directory_initializer import DirectoryInitializer


def main():
    directory = DirectoryInitializer(os.path.curdir)


if __name__ == '__main__':
    main()
