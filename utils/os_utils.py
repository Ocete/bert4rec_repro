import os
import subprocess
import shlex
import logging
import hashlib

# Change this to use these functions with its  behaviour, meant for Linux.
os_Windows = True
path_separator = '\\' if os_Windows else '/'

def split_cmd(cmd):
    cmd_splitted = cmd.split(' ') if os_Windows else shlex.split(cmd)
    return [ word for word in cmd_splitted if word != '' ]

def split_path(path):
    return path.split(path_separator)


def get_dir():
    utils_dirname = os.path.dirname(os.path.abspath(__file__))
    lib_dirname = os.path.abspath(os.path.join(utils_dirname, ".."))
    return lib_dirname

def recursive_listdir(dir_name):
    result = []
    for name in os.listdir(dir_name):
        full_name = os.path.join(dir_name, name)
        if(os.path.isdir(full_name)):
            result += recursive_listdir(full_name)
        else:
            result.append(full_name)
    return result
    
def shell(cmd):
    logging.info("running shell command: \n {}".format(cmd))
    cmd_splitted = split_cmd(cmd)
    if os_Windows and cmd_splitted[0] == 'unzip':
        cmd_splitted = ["tar", "-xvf", cmd_splitted[2], "-C", cmd_splitted[4]]
    if os_Windows and cmd_splitted[0] == 'mv':
        cmd_splitted[0] = 'move'
    if os_Windows and cmd_splitted[0] == 'rm':
        cmd_splitted[0] = 'del'
    subprocess.check_call(cmd_splitted)

def mkdir_p(dir_path):
    if not os.path.isdir(dir_path):
        shell("mkdir -p {}".format(dir_path))
        

def mkdir_p_local(relative_dir_path):
    """create folder inside of library if does not exists"""
    local_dir = get_dir()
    abspath = os.path.join(local_dir, relative_dir_path)
    mkdir_p(abspath)
    return abspath


def file_md5(fname):
    hash_md5 = hashlib.md5()
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def console_logging():
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
