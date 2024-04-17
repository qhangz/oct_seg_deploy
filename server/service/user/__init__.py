# 用户注册登录模块
import flask
import sqlite3
from service import utils


def register(app: flask.Flask, data_db: sqlite3.Connection):
    @app.route('/api/user/register', methods=['POST'])
    def register():
        username = flask.request.form['username']
        password = flask.request.form['password']
        email = flask.request.form['email']
        cursor = data_db.cursor()
        cursor.execute("SELECT * FROM user WHERE username = ?", (username,))
        result = cursor.fetchone()
        if result is not None:
            return utils.Resp(400, None, 'user already existed').to_json()
        cursor.execute("INSERT INTO user (username, password,email) VALUES (?, ?, ?)",
                       (username, password, email,))
        data_db.commit()
        return utils.Resp(200, None, 'register successfully').to_json()
    @app.route('/api/user/login', methods=['POST'])
    def login():
        username = flask.request.form['username']
        password = flask.request.form['password']
        cursor = data_db.cursor()
        cursor.execute("SELECT * FROM user WHERE username = ?", (username,))
        result = cursor.fetchone()
        if result is None:
            return utils.Resp(400, None, 'user not found').to_json()
        if str(result[2]).strip() == password.strip():
            return utils.Resp(200, None, 'login successfully').to_json()
        else:
            return utils.Resp(400, None, 'password error').to_json()


