from __future__ import print_function
import mimetypes
import os
import errno

# Python archiver
# Like shar, but in Python.  And no running of untrusted scripts!

EOF = '__EOF__'
EOF_LEN = len(EOF)

# http://stackoverflow.com/questions/273192/how-to-check-if-a-directory-exists-and-create-it-if-necessary
def make_sure_path_exists(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise

def nop(x):
    pass

def pack_archive(fout, files, report_fn=nop):
    for file in sorted(files):
        # Skip files that don't exist, no problem.
        if not os.path.exists(file):
            continue
        with open(file, 'r') as fin:
            report_fn(file)
            fout.write('======================== FILE %s\n' % file)
            fout.write('FILE: %s\n' % file)
            for line in fin:
                if line[:EOF_LEN] == EOF:
                    raise ValueError('Cannot pyar a file with a line starting in __EOF__')
                fout.write(line)
                # Make sure last line always ends with a newline
                if line[-1] != '\n':
                    fout.write('\n')
            fout.write(EOF + '\n')

def unpack_archive(fin, dest):
    fout = None
    try:
        state = 0    # 0 = looking for new file; 1 = in file
        for line in fin:
            if state == 0:
                if line[0:6] == 'FILE: ':
                    line = line[6:]
                    parts = line.strip().split(':')
                    fname = parts[0]    # Will be more parts later
                    fname = os.path.join(dest, fname)
                    dir = os.path.split(fname)[0]
                    make_sure_path_exists(dir)
                    # print('writing ', fname)
                    fout = open(fname, 'w')
                    state = 1
                else:
                    # Comment line between files
                    pass
            else:
                if line[:EOF_LEN] == EOF:
                    state = 0
                else:
                    fout.write(line)
    finally:
        if fout is not None:
            fout.close()


def list_archive(fin):
    fout = None
    state = 0    # 0 = looking for new file; 1 = in file
    for line in fin:
        if state == 0:
            if line[0:6] == 'FILE: ':
                line = line[6:]
                parts = line.strip().split(':')
                fname = parts[0]    # Will be more parts later
                yield fname
                state = 1
        else:
            if line[:EOF_LEN] == EOF:
                state = 0


if __name__ == '__main__':
    files = """./llnl2/util/link_tree.py
./llnl2/util/lang.py
./llnl2/util/lock.py""".split('\n')

    with open('test.pyar', 'w') as fout:
        pack_archive(fout, files)

    with open('test.pyar', 'r') as fin:
        unpack_archive(fin, 'xx')
