from flask import Flask, request, render_template, session
from jinja2 import Environment, PackageLoader
import os
from flask import redirect, url_for
import flask

os.system("pip install ultralytics")

from ultralytics import YOLO

# Load a pretrained YOLOv8n model
model = YOLO(r'C:\Users\l1kr1\OneDrive\Documents\project_ai_model_purblish\models\best.pt')

app = Flask(__name__)

# Создаем директорию для загрузки файлов, если она не существует
UPLOAD_FOLDER = r'C:\Users\l1kr1\OneDrive\Documents\project_ai_model_purblish\uploads'
ALLOWED_EXTENSIONS = {'mp4', "wav"}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/', methods=['GET', 'POST'])
def home():
    return render_template("home.html", label_text="")



@app.route('/result', methods=['GET', 'POST'])
def result():
    return render_template("result.html", report=request.args.get("report"))


def do_predict(video_path):
    # Run inference on video with arguments
    e_ = model.predict(video_path,
                save=True, imgsz=320, conf=0.95, vid_stride=60)
    a__ = e_[0].names
    print(a__)
    all_results = []
    for i, result in enumerate(e_):
        probs = result.probs  # Probs object for classification outputs
        all_results.append(a__[probs.top1])
    print(all_results)

    max_res_prob = "" 
    max_res_count = 0
    print(a__)
    for i in list(a__.values()):
        cur_count = all_results.count(i)
        if cur_count > max_res_count:
            max_res_count = cur_count
            max_res_prob = i
    print(max_res_prob, max_res_count)
    return max_res_prob, max_res_count, len(all_results)



@app.route('/file_send', methods=['GET', 'POST'])
def file_send():
    if request.method == 'POST':
        if 'load_file' not in request.files:
            print('No file part')
            return flask.redirect(flask.url_for("home"))
        file = request.files['load_file']
        if file:
            filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            print(filename)
            file.save(filename)
            max_res_prob, max_res_count, all_count= do_predict(filename)
            
            # Здесь вы можете добавить код для анализа файла и генерации отчета
            report = f"Мы обнаружили больше всего {max_res_prob}. {max_res_count} из {all_count} проверок."
            print(report)
            return flask.redirect(flask.url_for("result", report=report))
    


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5011)
