import flask
import re

# 统一响应格式
class Resp:
    def __init__(self, status_code, data=None, msg=None):
        self.status_code = status_code
        self.data = data
        self.msg = msg

    def to_json(self):
        return flask.jsonify({
            'code': self.status_code,
            'msg': self.msg,
            'data': self.data,
        })

# 去除image base64头
def remove_image_b64_header(b64):
    return re.sub('^data:image/.+;base64,', '', b64)