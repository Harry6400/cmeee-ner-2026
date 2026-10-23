"""Loopback-only teacher acceptance UI; live CPU/API work is separate from frozen replay."""
from __future__ import annotations
import argparse,json,mimetypes,re,signal,time
from http.server import ThreadingHTTPServer,BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse,parse_qs,unquote
from teacher_project import TeacherProject,BusyError,AssetError
from teacher_api import api_presets,APIError,public_diagnostic  # Snapshot settings before startup.
ROOT=Path(__file__).resolve().parents[1]


def handler_for(project):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args): pass
        def send(self,status,data,ctype='application/json; charset=utf-8',download=None):
            content=data if isinstance(data,bytes) else json.dumps(data,ensure_ascii=False).encode('utf-8')
            self.send_response(status); self.send_header('Content-Type',ctype); self.send_header('Content-Length',str(len(content)))
            self.send_header('Cache-Control','no-store'); self.send_header('X-Content-Type-Options','nosniff'); self.send_header('Referrer-Policy','no-referrer')
            if download: self.send_header('Content-Disposition',"attachment; filename*=UTF-8''"+__import__('urllib.parse',fromlist=['quote']).quote(download))
            self.end_headers()
            try: self.wfile.write(content)
            except (BrokenPipeError,ConnectionResetError): pass
        def _file(self,path,download=False):
            if path is None or not path.is_file(): return self.send(404,dict(error='文件尚未生成或不可用'))
            resolved=path.resolve()
            if not resolved.is_relative_to(project.root): return self.send(404,dict(error='文件不可用'))
            ctype=mimetypes.guess_type(path.name)[0] or 'application/octet-stream'
            if ctype.startswith('text/') or ctype in ('application/javascript','image/svg+xml'): ctype+='; charset=utf-8'
            return self.send(200,path.read_bytes(),ctype,path.name if download else None)
        def _path(self):
            parsed=urlparse(self.path); path=unquote(parsed.path)
            if '\\' in path or '%' in path or any(p in ('.','..') for p in path.split('/')): raise ValueError('非法路径')
            return path,parsed
        def do_GET(self):
            try:
                path,u=self._path()
                if path=='/': return self._file(project.root/'ui/stage5.html')
                if path=='/favicon.ico': return self.send(204,b'','image/x-icon')
                if re.fullmatch(r'/ui/[A-Za-z0-9_-]+\.(?:js|css)',path): return self._file(project.root/path.lstrip('/'))
                if re.fullmatch(r'/report-source/figures/[A-Za-z0-9_-]+\.(?:svg|png)',path): return self._file(project.root/path.lstrip('/'))
                if path=='/api/health':
                    try: usage=project.api().usage_summary()
                    except (OSError,ValueError): return self.send(503,dict(error='API用量记录不可读；请检查本地脱敏日志。',error_code='usage_unavailable'))
                    return self.send(200,dict(status='ready',runtime='local CPU',model_routes='live D0/N0/N1 CPU;separate historical replay',local_CPU_api_calls=0,teacher_API_calls=usage['calls'],teacher_API_calls_unknown=usage['unknown_calls'],teacher_API_usage=usage))
                if path=='/api/api-presets':
                    from teacher_api import api_presets
                    return self.send(200,api_presets())
                if path=='/api/project': return self.send(200,project.project())
                if path=='/api/cases': return self.send(200,project.cases())
                if path=='/api/case': return self.send(200,project.case(parse_qs(u.query).get('id',[''])[0]))
                if path=='/api/reproduction/jobs': return self.send(200,project.jobs.list())
                match=re.fullmatch(r'/api/reproduction/jobs/([A-Za-z0-9-]+)',path)
                if match:return self.send(200,project.jobs.get(match[1]))
                match=re.fullmatch(r'/artifacts/([a-z-]+)',path)
                if match:
                    item=project.artifact_paths().get(match[1]); return self._file(item[1],match[1]!='report') if item else self.send(404,dict(error='交付项不存在'))
                return self.send(404,dict(error='页面或接口不存在'))
            except KeyError as exc: return self.send(404,dict(error=str(exc.args[0])))
            except ValueError: return self.send(400,dict(error='请求格式或路径不合法'))
            except Exception as exc:
                project.record(event='http_error',method='GET',error_type=type(exc).__name__)
                return self.send(503,dict(error='服务暂不可用；请检查资产与本地日志。'))
        def do_POST(self):
            path=None
            host=self.headers.get('Host',''); allowed={f'127.0.0.1:{self.server.server_port}',f'localhost:{self.server.server_port}'}
            if host not in allowed or self.headers.get('Origin') not in {None,'http://'+host}: return self.send(403,dict(error='仅允许本服务本地同源请求'))
            try:
                path,_=self._path()
                if self.headers.get('Transfer-Encoding'): raise ValueError('不支持分块请求体')
                if self.headers.get('Content-Type','').split(';')[0].strip()!='application/json': raise ValueError('请求必须使用application/json')
                length=int(self.headers.get('Content-Length','0'))
                if not 0<length<=65536: raise ValueError('请求体为空或超过64KB')
                data=json.loads(self.rfile.read(length).decode('utf-8'))
                if not isinstance(data,dict): raise ValueError('请求必须是JSON对象')
                if path=='/api/extract': return self.send(200,project.extract(data))
                if path=='/api/api-check': return self.send(200,project.api().check(data.get('config',{})))
                if path=='/api/api-models': return self.send(200,project.api().models(data.get('config',{})))
                if path=='/api/extract-api': return self.send(200,project.api().extract(data))
                if path=='/api/reproduction/jobs':
                    if data: raise ValueError('复现任务不接受路径或命令参数')
                    return self.send(202,project.jobs.start())
                match=re.fullmatch(r'/api/reproduction/jobs/([A-Za-z0-9-]+)/cancel',path)
                if match:
                    if data: raise ValueError('取消任务不接受额外参数')
                    return self.send(200,project.jobs.cancel(match[1]))
                return self.send(404,dict(error='接口不存在'))
            except KeyError as exc:return self.send(404,dict(error=str(exc.args[0])))
            except BusyError as exc:return self.send(409,dict(error=str(exc)))
            except AssetError as exc:return self.send(503,dict(error=str(exc)))
            except APIError as exc:
                diagnostic=public_diagnostic(exc)
                project.record(event='api_error',path=path,error_type=type(exc).__name__,
                               error_code=diagnostic['error_code'],stage=diagnostic['stage'],
                               request_sent=diagnostic['request_sent'],http_status=diagnostic.get('http_status'))
                return self.send(503 if exc.code=='storage_not_writable' else 502,diagnostic)
            except (ValueError,TypeError,UnicodeDecodeError) as exc:
                local_errors={'未知抽取路线','请输入1—4000个Unicode字符。','请求必须使用application/json','不支持分块请求体','请求体为空或超过64KB','请求必须是JSON对象','复现任务不接受路径或命令参数','取消任务不接受额外参数'}
                if path in ('/api/api-check','/api/api-models','/api/extract-api'):
                    return self.send(400,public_diagnostic(exc))
                else: message=str(exc) if str(exc) in local_errors else '请求参数不合法；请检查文本、方法与配置。'
                return self.send(400,dict(error=message))
            except Exception as exc:
                # Exception text may contain a provider response or credentials; never echo it.
                project.record(event='http_error',method='POST',error_type=type(exc).__name__)
                if path in ('/api/api-check','/api/api-models','/api/extract-api'):
                    return self.send(503,public_diagnostic(exc))
                return self.send(503,dict(error='本次操作未完成；请检查配置或本地日志。'))
    return Handler


def make_server(root=ROOT,port=8765,log_dir=None,project=None):
    project=project or TeacherProject(root,log_dir)
    server=ThreadingHTTPServer(('127.0.0.1',port),handler_for(project));server.daemon_threads=True;server.project=project
    return server

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--port',type=int,default=8765);parser.add_argument('--log-dir',type=Path);args=parser.parse_args()
    started=time.perf_counter();server=make_server(port=args.port,log_dir=args.log_dir);project=server.project
    project.record(event='server_start',port=server.server_port,load_seconds=time.perf_counter()-started,new_API_calls=0,GPU_started=False,lazy_assets=True)
    print(f'教师验收页面已启动：http://127.0.0.1:{server.server_port}',flush=True)
    def stop(signum,frame): raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,stop)
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:project.close();server.server_close();project.record(event='server_stop')
if __name__=='__main__':main()
