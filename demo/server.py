#!/usr/bin/env python
##
##  DerpYOLO web demo
##
##  $ openssl req -newkey rsa:2048 -nodes -keyout key.pem -x509 -days 365 -out cert.pem
##  $ python server.py -s :4343
##
import sys
import os
import re
import cgi
import traceback
import ssl
import base64
import json
from io import BytesIO
from PIL import Image

sys.path.append('..')
import torch
from model import YOLONet
from categories import CATEGORIES
from detect import init_model, detect, softnms
from objutils import rect_fit, rect_map, adjust_image

# quote HTML metacharacters.
def q(s):
    assert isinstance(s, str), s
    return (s.
            replace('&','&amp;').
            replace('>','&gt;').
            replace('<','&lt;').
            replace('"','&#34;').
            replace("'",'&#39;'))

# encode as a URL.
URLENC = re.compile(r'[^a-zA-Z0-9_.-]')
def urlenc(url, codec='utf-8'):
    def f(m):
        return '%%%02X' % ord(m.group(0))
    return URLENC.sub(f, url.encode(codec))

# remove redundant spaces.
RMSP = re.compile(r'\s+', re.U)
def rmsp(s):
    return RMSP.sub(' ', s.strip())

# merge two dictionaries.
def mergedict(d1, d2):
    d1 = d1.copy()
    d1.update(d2)
    return d1

# iterable
def iterable(obj):
    return hasattr(obj, '__iter__')

# closable
def closable(obj):
    return hasattr(obj, 'close')


##  Template
##
class Template(object):

    def __init__(self, *args, **kwargs):
        if '_copyfrom' in kwargs:
            _copyfrom = kwargs['_copyfrom']
            objs = _copyfrom.objs
            kwargs = mergedict(_copyfrom.kwargs, kwargs)
        else:
            objs = []
            for line in args:
                if not isinstance(line, str):
                    raise ValueError('Non-string object in a template: %r' % (args,))
                i0 = 0
                for m in self._VARIABLE.finditer(line):
                    objs.append(line[i0:m.start(0)])
                    x = m.group(1)
                    if x == '$':
                        objs.append(x)
                    else:
                        objs.append(self.Variable(x[0], x[1:-1]))
                    i0 = m.end(0)
                objs.append(line[i0:])
        self.objs = objs
        self.kwargs = kwargs
        return

    def __call__(self, **kwargs):
        return self.__class__(_copyfrom=self, **kwargs)

    def __iter__(self):
        return self.render()

    def __repr__(self):
        return ('<Template %r>' % self.objs)

    def __str__(self):
        return ''.join(self)

    @classmethod
    def load(klass, lines, **kwargs):
        template = klass(*lines, **kwargs)
        if closable(lines):
            lines.close()
        return template

    def render(self, codec='utf-8', **kwargs):
        kwargs = mergedict(self.kwargs, kwargs)
        def render1(value, quote=False):
            if value is None:
                pass
            elif isinstance(value, Template):
                if quote:
                    raise ValueError('Template in a quoted context: %r' % self)
                else:
                    for x in value.render(codec=codec, **kwargs):
                        yield x
            elif isinstance(value, dict):
                raise ValueError('Dictionary not allowed: %r' % self)
            elif isinstance(value, bytes):
                yield value
            elif isinstance(value, str):
                if quote:
                    yield q(value)
                else:
                    yield value
            elif callable(value):
                for x in render1(value(**kwargs), quote=quote):
                    yield x
            elif iterable(value):
                for obj1 in value:
                    for x in render1(obj1, quote=quote):
                        yield x
            elif quote:
                yield q(str(value))
            else:
                raise ValueError('Non-string object in a non-quoted context: %r' % self)
            return
        for obj in self.objs:
            if isinstance(obj, self.Variable):
                k = obj.name
                if k in kwargs:
                    value = kwargs[k]
                elif k in self.kwargs:
                    value = self.kwargs[k]
                else:
                    raise ValueError('Parameter not found: %r in %r' % (k, self))
                if obj.type == '(':
                    for x in render1(value, quote=True):
                        yield x
                    continue
                elif obj.type == '[':
                    yield urlenc(value)
                    continue
            else:
                value = obj
            for x in render1(value):
                yield x
        return

    _VARIABLE = re.compile(r'\$(\(\w+\)|\[\w+\]|<\w+>)')

    class Variable(object):

        def __init__(self, type, name):
            self.type = type
            self.name = name
            return

        def __repr__(self):
            if self.type == '(':
                return '$(%s)' % self.name
            elif self.type == '[':
                return '$[%s]' % self.name
            else:
                return '$<%s>' % self.name


##  Router
##
class Router(object):

    def __init__(self, method, regex, func):
        self.method = method
        self.regex = regex
        self.func = func
        return

    @staticmethod
    def make_wrapper(method, pat):
        regex = re.compile('^'+pat+'$')
        def wrapper(func):
            return Router(method, regex, func)
        return wrapper

def HEAD(pat): return Router.make_wrapper('HEAD', pat)
def GET(pat): return Router.make_wrapper('GET', pat)
def POST(pat): return Router.make_wrapper('POST', pat)


##  Response
##
class Response(object):

    def __init__(self, status='200 OK', content_type='text/html', **kwargs):
        self.status = status
        self.headers = [('Content-Type', content_type)]+list(kwargs.items())
        return

    def add_header(self, k, v):
        self.headers.append((k, v))
        return

class Redirect(Response):

    def __init__(self, location):
        Response.__init__(self, '302 Found', Location=location)
        return

class NotFound(Response):

    def __init__(self):
        Response.__init__(self, '404 Not Found')
        return

class InternalError(Response):

    def __init__(self):
        Response.__init__(self, '500 Internal Server Error',
                          content_type='text/plain')
        return


##  WebApp
##
class WebApp(object):

    debug = 0
    codec = 'utf-8'

    def run(self, environ, start_response):
        method = environ.get('REQUEST_METHOD', 'GET')
        path = environ.get('PATH_INFO', '/')
        fp = environ.get('wsgi.input')
        fields = cgi.FieldStorage(fp=fp, environ=environ)
        router = kwargs = None
        for attr in dir(self):
            router = getattr(self, attr)
            if not isinstance(router, Router): continue
            if router.method != method: continue
            m = router.regex.match(path)
            if m is None: continue
            params = m.groupdict().copy()
            params['_path'] = path
            params['_fields'] = fields
            params['_environ'] = environ
            code = router.func.__code__
            args = code.co_varnames[:code.co_argcount]
            kwargs = {}
            for k in args[1:]:
                if k in fields:
                    kwargs[k] = fields.getvalue(k)
                elif k in params:
                    kwargs[k] = params[k]
            break
        def expn(obj):
            if (isinstance(obj, Response) or
                isinstance(obj, str) or
                isinstance(obj, bytes)):
                yield obj
            elif isinstance(obj, Template):
                for x in obj.render(codec=self.codec):
                    if isinstance(x, str):
                        x = x.encode(self.codec)
                    yield x
            elif iterable(obj):
                for x in obj:
                    for y in expn(x):
                        yield y
            else:
                yield obj
            return
        resp = []
        try:
            if kwargs is None:
                result = self.get_default(path, fields, environ)
            else:
                result = router.func(self, **kwargs)
            for obj in expn(result):
                if isinstance(obj, Response):
                    if resp is None:
                        raise ValueError('Multiple Responses: %r' % obj)
                    start_response(obj.status, obj.headers)
                    for x in resp:
                        yield x
                    resp = None
                else:
                    if isinstance(obj, bytes):
                        pass
                    elif isinstance(obj, str):
                        obj = obj.encode(self.codec)
                    else:
                        obj = str(obj).encode(self.codec)
                    if resp is None:
                        yield obj
                    else:
                        resp.append(obj)
            if resp is not None:
                obj = InternalError()
                start_response(obj.status, obj.headers)
                if self.debug:
                    yield ('No response: %r' % resp).encode('utf-8')
                else:
                    sys.stderr.write('No response: %r\n' % resp)
        except Exception as e:
            if resp is not None:
                obj = InternalError()
                start_response(obj.status, obj.headers)
            tb = traceback.format_exc()
            if self.debug:
                yield tb.encode('utf-8')
            else:
                sys.stderr.write('Error: %s\n' % tb)
        return

    def get_default(self, path, fields, environ):
        return [NotFound(), '<html><body>not found</body></html>\n']


# run_server
def run_server(host, port, app):
    from wsgiref.simple_server import make_server
    print('Serving on %r port %d...' % (host, port))
    httpd = make_server(host, port, app.run)
    httpd.socket = ssl.wrap_socket(
        httpd.socket, keyfile='key.pem', certfile='cert.pem', server_side=True)
    httpd.serve_forever()

# run_cgi
def run_cgi(app):
    from wsgiref.handlers import CGIHandler
    CGIHandler().run(app.run)

# run_httpcgi: for cgi-httpd
def run_httpcgi(app):
    from wsgiref.handlers import CGIHandler
    class HTTPCGIHandler(CGIHandler):
        def start_response(self, status, headers, exc_info=None):
            protocol = self.environ.get('SERVER_PROTOCOL', 'HTTP/1.0')
            sys.stdout.write('%s %s\r\n' % (protocol, status))
            return CGIHandler.start_response(self, status, headers, exc_info=exc_info)
    HTTPCGIHandler().run(app.run)

# main
def main(app, argv):
    import getopt
    def usage():
        print('usage: %s [-d] [-s [host]:port] [method [url]]' % argv[0])
        return 100
    try:
        (opts, args) = getopt.getopt(argv[1:], 'ds:')
    except getopt.GetoptError:
        return usage()
    host = ''
    port = 4343
    debug = 0
    for (k, v) in opts:
        if k == '-d': debug += 1
        elif k == '-s':
            (host,_,port) = v.partition(':')
    WebApp.debug = debug
    if port is not None:
        run_server(host, int(port), app)
    else:
        if args:
            os.environ['REQUEST_METHOD'] = args.pop(0)
        if args:
            url = args.pop(0)
            (path,_,qs) = url.partition('?')
            os.environ['PATH_INFO'] = path
            os.environ['QUERY_STRING'] = qs
        run_cgi(app)
    return


##  Sample App
##
class SampleApp(WebApp):

    def __init__(self):
        WebApp.__init__(self)
        (self.model, self.max_objs) = init_model('../yolo_net.pt')
        self.threshold = 0.20
        return

    @GET('/')
    def index(self):
        yield Response()
        with open('client.html') as fp:
            data = fp.read()
        yield data
        return

    @POST('/detect')
    def yolo(self, data):
        yield Response()
        result = []
        if data.startswith('data:image/jpeg;base64,'):
            (_,_,data) = data.partition(',')
            data = base64.urlsafe_b64decode(data)
            image = Image.open(BytesIO(data))
            (window, frame) = rect_fit(YOLONet.IMAGE_SIZE, image.size)
            tmpimage = adjust_image(image, window, YOLONet.IMAGE_SIZE)
            found = detect(self.model, tmpimage, max_objs=self.max_objs)
            found = softnms(found, self.threshold)
            print(image, found)
            result = [
                {
                    'name':obj.name,
                    'color':obj.color,
                    'bbox':rect_map(frame, YOLONet.IMAGE_SIZE, obj.bbox)
                } for obj in found
            ]
        yield Template(json.dumps(result))
        return

if __name__ == '__main__': sys.exit(main(SampleApp(), sys.argv))
