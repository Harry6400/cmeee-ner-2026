"""Terminate only the subprocess tree created by the local reproduction service."""
import os
import signal
import subprocess
import time


class WindowsJob:
    """Windows Job Object; children inherit membership, last handle closes the tree."""
    def __init__(self):
        import ctypes
        from ctypes import wintypes as w
        class BasicLimits(ctypes.Structure):
            _fields_ = [('ProcessTime', ctypes.c_int64), ('JobTime', ctypes.c_int64),
                        ('Flags', w.DWORD), ('MinWorkingSet', ctypes.c_size_t),
                        ('MaxWorkingSet', ctypes.c_size_t), ('ActiveLimit', w.DWORD),
                        ('Affinity', ctypes.c_size_t), ('Priority', w.DWORD), ('Scheduling', w.DWORD)]
        class IOCounters(ctypes.Structure):
            _fields_ = [(name, ctypes.c_uint64) for name in
                        ('ReadOps', 'WriteOps', 'OtherOps', 'ReadBytes', 'WriteBytes', 'OtherBytes')]
        class ExtendedLimits(ctypes.Structure):
            _fields_ = [('Basic', BasicLimits), ('IO', IOCounters),
                        ('ProcessMemory', ctypes.c_size_t), ('JobMemory', ctypes.c_size_t),
                        ('PeakProcessMemory', ctypes.c_size_t), ('PeakJobMemory', ctypes.c_size_t)]
        class Accounting(ctypes.Structure):
            _fields_ = [(name, ctypes.c_int64) for name in ('User', 'Kernel', 'PeriodUser', 'PeriodKernel')] + [
                (name, w.DWORD) for name in ('PageFaults', 'TotalProcesses', 'ActiveProcesses', 'TerminatedProcesses')]
        self.ctypes = ctypes
        self.Accounting = Accounting
        self.api = ctypes.WinDLL('kernel32', use_last_error=True)
        signatures = {
            'CreateJobObjectW': ([ctypes.c_void_p, w.LPCWSTR], w.HANDLE),
            'SetInformationJobObject': ([w.HANDLE, ctypes.c_int, ctypes.c_void_p, w.DWORD], w.BOOL),
            'AssignProcessToJobObject': ([w.HANDLE, w.HANDLE], w.BOOL),
            'QueryInformationJobObject': ([w.HANDLE, ctypes.c_int, ctypes.c_void_p, w.DWORD, ctypes.c_void_p], w.BOOL),
            'TerminateJobObject': ([w.HANDLE, w.UINT], w.BOOL),
            'CloseHandle': ([w.HANDLE], w.BOOL),
        }
        for name, (args, result) in signatures.items():
            func = getattr(self.api, name); func.argtypes = args; func.restype = result
        self.handle = self.api.CreateJobObjectW(None, None)
        if not self.handle:
            raise ctypes.WinError(ctypes.get_last_error())
        limits = ExtendedLimits(); limits.Basic.Flags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not self.api.SetInformationJobObject(self.handle, 9, ctypes.byref(limits), ctypes.sizeof(limits)):
            error = ctypes.WinError(ctypes.get_last_error()); self.close(); raise error

    def assign(self, process):
        if not self.api.AssignProcessToJobObject(self.handle, int(process._handle)):
            raise self.ctypes.WinError(self.ctypes.get_last_error())

    def terminate(self):
        if not self.api.TerminateJobObject(self.handle, 1):
            raise self.ctypes.WinError(self.ctypes.get_last_error())

    def gone(self):
        info = self.Accounting()
        if not self.api.QueryInformationJobObject(self.handle, 1, self.ctypes.byref(info), self.ctypes.sizeof(info), None):
            return False
        return info.ActiveProcesses == 0

    def close(self):
        if self.handle:
            self.api.CloseHandle(self.handle); self.handle = None


def spawn(command, popen=subprocess.Popen, **kwargs):
    if os.name != 'nt':
        return popen(command, start_new_session=True, **kwargs)
    job = WindowsJob()
    process = None
    # Wait for assignment before executing any project code or spawning descendants.
    bootstrap = "import sys,runpy,os; token=sys.stdin.buffer.read(1); sys.argv=sys.argv[1:]; token==b'1' or sys.exit(125); sys.path.insert(0,os.path.dirname(os.path.abspath(sys.argv[0]))); runpy.run_path(sys.argv[0],run_name='__main__')"
    # Caller uses [python, -B, -u, script, ...]. Preserve the script and all arguments.
    wrapped = [command[0], '-X', 'utf8', '-B', '-u', '-c', bootstrap, *command[3:]]
    try:
        process = popen(wrapped, stdin=subprocess.PIPE, creationflags=0x200, **kwargs)
        job.assign(process); process._owned_job = job
        process.stdin.write('1' if kwargs.get('text') else b'1'); process.stdin.flush(); process.stdin.close()
        return process
    except BaseException:
        try:
            if process is not None:
                try:
                    job.terminate()
                finally:
                    if process.poll() is None: process.kill()
                    process.wait(timeout=5)
                    if process.stdin: process.stdin.close()
                    if process.stdout: process.stdout.close()
        finally:
            job.close()
        raise


def terminate(process, force=False):
    if os.name == 'nt':
        process._owned_job.terminate()
    else:
        try: os.killpg(process.pid, signal.SIGKILL if force else signal.SIGTERM)
        except ProcessLookupError: pass


def gone(process):
    if process is None: return True
    deadline = time.monotonic() + 1
    while True:
        if os.name == 'nt':
            absent = process._owned_job.gone()
        else:
            try: os.killpg(process.pid, 0); absent = False
            except ProcessLookupError: absent = True
            except PermissionError: absent = False
        if absent: return True
        if time.monotonic() >= deadline: return False
        time.sleep(.025)


def release(process):
    if process is not None and hasattr(process, '_owned_job'):
        process._owned_job.close()
