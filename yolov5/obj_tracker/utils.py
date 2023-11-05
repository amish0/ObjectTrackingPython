from pathlib import Path

def check_suffix(file='yolov8n.pt', suffix='.pt', msg=''):
    """Check file for acceptable suffix."""
    # check suffix is empty or not
    if suffix and file:
        # check suffix is string or not
        if isinstance(suffix, str) and isinstance(file, str):
            # check file ends with suffix or not
            return file.endswith(suffix)
        else:
            print(f'{msg}suffix and file must be string')
            return False
    else:
        print(f'{msg}suffix and file must not be empty')
        return False
        

def check_file(file='yolov8n.pt'):
    if file:
        if isinstance(file, str):
            # check file exists or not
            return Path(file).exists()
        else:
            print('file must be string')
            return False
    else:
        print('file must not be empty')
        return False
