"""Nonblocking process-exclusive lock for an already-open local file descriptor."""
import errno
import os

if os.name == 'nt':
    import msvcrt
else:
    import fcntl


def _windows_lock(fd, mode):
    position = os.lseek(fd, 0, os.SEEK_CUR)
    try:
        os.lseek(fd, 0, os.SEEK_SET)
        msvcrt.locking(fd, mode, 1)
    finally:
        os.lseek(fd, position, os.SEEK_SET)


def acquire(fd):
    if os.name != 'nt':
        return fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        _windows_lock(fd, msvcrt.LK_NBLCK)
    except OSError as exc:
        if exc.errno in (errno.EACCES, errno.EAGAIN, errno.EDEADLK):
            raise BlockingIOError(exc.errno, 'Another process owns the file lock') from exc
        raise


def release(fd):
    if os.name == 'nt':
        _windows_lock(fd, msvcrt.LK_UNLCK)
    else:
        fcntl.flock(fd, fcntl.LOCK_UN)
