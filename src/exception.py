import sys
import os

def get_exception_message(e):
    exc_type, exc_obj, exc_tb = sys.exc_info()
    fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]

    error_message = "Exception: " + str(e) + " at file " + fname + " in line " + str(exc_tb.tb_lineno)
    return error_message 

class CustomException(Exception):
    def __init__(self, message):
        super().__init__(message)
        self.message = message
        self.error_message = get_exception_message(message)

    def __str__(self):
            return self.error_message