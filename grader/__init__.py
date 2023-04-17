from .autograder_tokenizer import *

import typing
class AutograderSubmitter:
    def __init__(self):
        self.submission_data : typing.Dict[str, np.ndarray]  = {

        }
    
    def generate_submission_file(self, filename):
        np.savez_compressed(filename, **self.submission_data)