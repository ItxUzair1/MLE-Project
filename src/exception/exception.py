import sys


def get_detailed_message(error,error_detail:sys):
    _,_,exec_tb=error_detail.exc_info()
    
    file_name=exec_tb.tb_frame.f_code.co_filename
    line_number=exec_tb.tb_lineno
    
    error_message= (f"Error in file [{file_name}]"
                    f"at line no {line_number}"
                    f"and the error is {str(error)}")
    
    return error_message


class CustomException(Exception):
    def __init__(self,error,error_detail:sys):
        super().__init__(error)
        self.error=get_detailed_message(error,error_detail)
        
    
    def __str__(self):
        return self.error
    
    