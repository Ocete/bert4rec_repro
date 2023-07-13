import os
import subprocess
import shlex
import logging
import hashlib

# Change this to use these functions with its original behaviour,
# meant for Linux.
os_Windows = True
path_separator = '\\' if os_Windows else '/'
python_exec_path = 'D:\\anaconda3\\envs\\aprec_repro\\python.exe'

def prepare_cmd_call(cmd):
    cmd_splitted = cmd.split(' ') if os_Windows else shlex.split(cmd)
    cmd_splitted = [ word for word in cmd_splitted if word != '' ]
    if os_Windows and cmd_splitted[0] == 'unzip':
        cmd_splitted = ["tar", "-xvf", cmd_splitted[2], "-C", cmd_splitted[4]]
    if os_Windows and cmd_splitted[0] == 'mv':
        cmd_splitted[0] = 'move'
    if os_Windows and cmd_splitted[0] == 'rm':
        cmd_splitted[0] = 'del'
    if os_Windows and (cmd_splitted[0] == 'python' or cmd_splitted[0] == 'python3'):
        cmd_splitted[0] = python_exec_path
    return cmd_splitted

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
    cmd_splitted = prepare_cmd_call(cmd)
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
