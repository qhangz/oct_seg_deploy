import sqlite3
import os
from flask import Flask
import base64

app = Flask(__name__)

DATABASE = 'data.db'

def create_database():
    try:
        # 检查数据库文件是否存在
        if not os.path.exists(DATABASE):
            db = sqlite3.connect(DATABASE)
            print("Opened database successfully")
            # 从 schema.sql 文件中读取 SQL 命令并执行
            with app.open_resource('schema.sql', mode='r') as f:
                db.cursor().executescript(f.read())
            print("Table created successfully")
            # 向data.db中插入annotations
            # 读取当前文件夹下的annotations文件夹中的图片，并以base64格式存入数据库
            # annotations_folder = './annotations'
            # # print('annotations_folder:',annotations_folder)
            # for filename in os.listdir(annotations_folder):
            #     if filename.endswith('.jpg') or filename.endswith('.png'):  # 只处理图片文件
            #         with open(os.path.join(annotations_folder, filename), 'rb') as img_file:
            #             img_data = base64.b64encode(img_file.read()).decode('utf-8')  # 转换为base64编码并解码为字符串
            #             db.cursor().execute("INSERT INTO output (imgname, img) VALUES (?, ?)",
            #                                 (filename, img_data))
            #             # print(f"{filename} inserted into database")

            # 提交并关闭数据库连接
            # db.commit()

            db.close()
        else:
            print("Database file already exists, skipping creation")
    except Exception as e:
        print("Error:",e)

if __name__ == '__main__':
    create_database()