import sys
import logging

def generate_error_message(error, exception_info:sys):
    """
    This function displays an error message with file name and line number.
    Args:
        error (Exception): The error or exception object.
        exception_info (tuple): The result of sys.exc_info().
    Returns:
        str: A formatted error message.
    """
    try:
        _,_,exc_tb=exception_info.exc_info()
        file_name=exc_tb.tb_frame.f_code.co_filename
        error_message="An error was found in [{0}], specifically in line [{1}] with error message [{2}].".format(
        file_name,exc_tb.tb_lineno,str(error))
        return error_message
    except Exception as e:
        return f"Error occurred while processing error: {str(e)}"

class CustomException(Exception):
    """
    This is a custom exception class for encapsulating error messages and details.
    Args:
        error_message (str): A descriptive error message.
        exception_info (sys (any)): Additional details about the error (name, row, file, etc.)

    Attributes:
        error_detail (Optional[Any]): The optional error detail.

    """
    def __init__(self, error_message, exception_info:sys):
        super().__init__(error_message)
        self.error_message = generate_error_message(error_message, exception_info=exception_info)

    def __str__(self):
        return self.error_message

if __name__=="__main__":

    try:
        a=1/0
    except Exception as e:
        logging.info("divide by 0")
        raise CustomException(e, sys)