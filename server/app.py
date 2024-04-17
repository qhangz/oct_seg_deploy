from flask import Flask
import sqlite3
from service.data import create_database
import service.images
import service.user
import os
import flask_cors

app = Flask(__name__)
# CORS
flask_cors.CORS(app,supports_credentials=True)

# 保存原始的工作目录
original_dir = os.getcwd()
# 切换到 ./service/data 文件夹下
os.chdir('./service/data')
# 调用创建数据库的函数
create_database()
# 恢复到原始的工作目录
os.chdir(original_dir)

data_db = sqlite3.connect('./service/data/data.db', check_same_thread=False)
# 设置为可多线程访问
data_db.isolation_level = None


# 服务注册
service.images.register(app, data_db)
service.user.register(app, data_db)

@app.route("/")
def index():
    return 'Index Page'


@app.route("/api/hello", methods=['GET', 'POST'])
def hello():
    return 'Hello, World'



if __name__ == '__main__':
    app.run()
