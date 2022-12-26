import traceback
import paddle
import torch
import os.path as osp

def _is_system_package(filename):
    exclude = [
        'lib/python', '/usr/local',
        osp.dirname(paddle.__file__),
        osp.dirname(torch.__file__),
        osp.dirname(__file__), # exclude PPAutoDiff
    ]
    for pattern in exclude:
        if pattern in filename:
            return True
    return False

def print_frame(f, indent=8):
    indent = ' '*indent
    print ("{} File {}: {}    {}\n{}{}{}".format(indent, f.filename, f.lineno, f.name, indent, indent, f.line))

def print_frames(fs, indent=8):
    for f in fs: 
        print_frame(f, indent)

def extract_frame_summary(): 
    """
    extract the current call stack by traceback module.
    gather the call information and put them into ReportItem to helper locate the error.

    frame_summary: 
        line: line of the code
        lineno: line number of the file
        filename: file name of the stack
        name: the function name.
    """
    frame_summarys = traceback.StackSummary.extract(traceback.walk_stack(None))
    last_user_fs = None
    for fs in frame_summarys: 
        if not _is_system_package(fs.filename): 
            last_user_fs = fs
            break
    assert last_user_fs is not None, "Error happend, can't return None."
    return last_user_fs, frame_summarys
