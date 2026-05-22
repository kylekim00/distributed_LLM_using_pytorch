import argparse

class MyParser(argparse.ArgumentParser):
    def __init__(self):
        self.parser = super().__init__()

        
        self.add_argument(
            "--download",
            action="store_true",
            help="model download"
        )
        