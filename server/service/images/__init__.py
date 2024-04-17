# 图片上传、存储、获取、删除
import base64
import sqlite3
import flask
from service import utils
from PIL import Image
import io
from io import BytesIO
import model.main

def register(app: flask.Flask, data_db: sqlite3.Connection):
    @app.route('/api/uploadimg', methods=['POST'])
    def upload():
        # 获取上传的文件
        uploaded_file = flask.request.files['file']

        # 读取文件内容
        file_content = uploaded_file.read()

        # 获取上传的文件名
        filename = uploaded_file.filename
        # print('filename:', filename)

        # 将文件内容转换为 Base64 格式
        base64_content = base64.b64encode(file_content).decode('utf-8')

        # 将 Base64 编码后的内容存储到数据库中
        cursor = data_db.cursor()
        cursor.execute("INSERT INTO imgdata (inputname, inputimg) VALUES (?, ?)",
                       (filename, base64_content,))
        data_db.commit()

        # 获取刚插入的图像的 ID
        image_id = cursor.lastrowid

        return utils.Resp(200, {'image_id': image_id}, 'upload successfully').to_json()

    @app.route('/api/getimg/<int:image_id>', methods=['GET'])
    def get_image(image_id):
        # 从数据库中获取图像数据
        cursor = data_db.cursor()
        cursor.execute("SELECT inputimg FROM imgdata WHERE id = ?", (image_id,))
        result = cursor.fetchone()

        if result is None:
            return utils.Resp(404, None, 'image not found').to_json()

        # 获取图像的 Base64 编码内容
        base64_content = result[0]

        # 将 Base64 编码的图像数据转换为图像格式
        image_data = base64.b64decode(base64_content)

        # 将图像数据转换为 BytesIO 对象
        image_stream = BytesIO(image_data)

        # 将 BytesIO 对象转换为 PIL 图像对象
        image = Image.open(image_stream)

        # 调用图像分割和标注函数(字节流格式返回)
        result_img_bytes = model.main.seg(image)

        # 将字节流格式的图像转换为 base64 字符串
        result_img_base64 = base64.b64encode(result_img_bytes).decode('utf-8')
        # 将 Base64 编码后的内容根据image_id存储到数据库中
        cursor.execute("UPDATE imgdata SET outputimg = ? WHERE id = ?", (result_img_base64, image_id,))
        data_db.commit()


        # 将处理好的图像转换为字节流并返回
        return flask.send_file(
            io.BytesIO(result_img_bytes),
            mimetype='image/png',
            as_attachment=True,
            # attachment_filename='result_image.png'
            download_name='result_image.png'
        )

