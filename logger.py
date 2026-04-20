import os
import sys
from datetime import datetime


class TeeStream:
    """同时输出到控制台和文件的流对象"""

    def __init__(self, original_stream, file_handle):
        self.original_stream = original_stream
        self.file_handle = file_handle
        self.buffer = ""

    def write(self, msg):
        self.original_stream.write(msg)
        self.file_handle.write(msg)

    def flush(self):
        self.original_stream.flush()
        self.file_handle.flush()

    def fileno(self):
        return self.original_stream.fileno()

    def isatty(self):
        return self.original_stream.isatty()

    def read(self, size=-1):
        return self.original_stream.read(size)

    def readline(self, size=-1):
        return self.original_stream.readline(size)

    def readlines(self, hint=-1):
        return self.original_stream.readlines(hint)

    def seek(self, offset, whence=0):
        return self.original_stream.seek(offset, whence)

    def tell(self):
        return self.original_stream.tell()

    def truncate(self, size=None):
        return self.original_stream.truncate(size)


class Logger:
    """日志记录器 - 自动捕获所有print输出"""

    def __init__(self, log_dir='./logs', date_str=None):
        self.log_dir = log_dir
        self.date_str = date_str or datetime.now().strftime('%Y-%m-%d')
        self.log_file = os.path.join(self.log_dir, f"{self.date_str}.md")
        self.start_time = datetime.now()
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        self._file_handle = None
        self._tee_stream = None

    def start(self):
        os.makedirs(self.log_dir, exist_ok=True)
        self._file_handle = open(self.log_file, 'w', encoding='utf-8')

        header = f"""# 量化选股系统运行日志

## 运行信息
- **运行日期**: {self.date_str}
- **运行时间**: {self.start_time.strftime('%H:%M:%S')}
- **日志文件**: {os.path.basename(self.log_file)}

---

## 运行输出

"""
        self._file_handle.write(header)
        self.original_stdout.write(header)

        self._tee_stream = TeeStream(self.original_stdout, self._file_handle)
        sys.stdout = self._tee_stream

        return self

    def finish(self):
        end_time = datetime.now()
        duration = end_time - self.start_time

        sys.stdout = self.original_stdout

        footer = f"""

---

## 运行统计
- **结束时间**: {end_time.strftime('%H:%M:%S')}
- **总耗时**: {duration.total_seconds():.2f} 秒

"""
        self._file_handle.write(footer)
        self.original_stdout.write(footer)
        self._file_handle.close()

        self.original_stdout.write(f"\n✅ 日志已保存到: {self.log_file}\n")

    def section(self, title):
        sep = "=" * 60
        print(f"\n{sep}")
        print(f"  {title}")
        print(f"{sep}\n")

    def subsection(self, title):
        sep = "-" * 40
        print(f"\n{sep}")
        print(f"  {title}")
        print(f"{sep}\n")


_log_instance = None

def get_logger(log_dir='./logs', date_str=None):
    global _log_instance
    if _log_instance is None:
        _log_instance = Logger(log_dir, date_str)
    return _log_instance

def init_logger(log_dir='./logs', date_str=None):
    global _log_instance
    _log_instance = Logger(log_dir, date_str)
    return _log_instance.start()

def close_logger():
    global _log_instance
    if _log_instance is not None:
        _log_instance.finish()
        _log_instance = None