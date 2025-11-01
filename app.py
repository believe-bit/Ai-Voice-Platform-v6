from flask import Flask, request, jsonify, Response, send_from_directory
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import os
import re
import subprocess
import json
import uuid
from werkzeug.utils import secure_filename
import zipfile
import shutil
from pathlib import Path
from threading import Thread
import queue
import time
import signal
import sys
import sqlite3
from datetime import datetime, timedelta
from contextlib import contextmanager
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import simpleSplit
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import logging
import jwt
from functools import wraps
from passlib.hash import bcrypt
import threading
import hashlib
from socketio import Client
import socket

# ==================== 全局配置 ====================
STUDENT_IPS_FILE = Path(__file__).with_name('User') / 'student_ips.txt'

app = Flask(__name__)
CORS(app)

# 使用 flask_socketio 启动 Flask 服务（关键）
socketio = SocketIO(app, cors_allowed_origins="*", logger=True)
# CORS(app, resources={r"/api/*": {"origins": "*"}})
# socketio = SocketIO(app, cors_allowed_origins="http://192.168.1.124:8082",logger=True)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models", "ASR_train_models")
UPLOAD_DIR = os.path.join(BASE_DIR, "Uploads")
OUTPUT_DIR = os.path.join(BASE_DIR, "VITS-fast-fine-tuning", "output")
VITS_BASE = os.path.join(BASE_DIR, 'VITS-fast-fine-tuning', 'dataset')
CLEANERS_FILE = os.path.join(BASE_DIR, 'VITS-fast-fine-tuning', 'text', 'cleaners.py')
CONFIG_DIR = os.path.join(BASE_DIR, 'VITS-fast-fine-tuning', 'configs')
ALLOWED_EXTENSIONS = {'zip'}
OFFLINE_MODEL_DIR = os.path.join(BASE_DIR, "models", "ASR_models")
OFFLINE_UPLOAD_DIR = os.path.join(BASE_DIR, "Uploads", "offline_audio")
ALLOWED_AUDIO_EXTENSIONS = {'wav'}
VITS_INFERENCE_DIR = os.path.join(BASE_DIR, 'VITS-fast-fine-tuning')
ALLOWED_MODEL_EXTENSIONS = {'pth'}
ALLOWED_CONFIG_EXTENSIONS = {'json'}
GUIDE_DIR = os.path.join(BASE_DIR, "Guides")
USER_DATA_ROOT = Path("/data/huangtianle/Ai-Voice-Platform/User")
# USERS_FILE = os.path.join(BASE_DIR, "users.json")  # 用户 JSON 文件

STREAM_ASR_PROC = None          # 子进程句柄
STREAM_ASR_LOCK = threading.Lock()

# 类型映射：英文 → 中文
TYPE_MAP = {
    "Speech Recognition": "语音识别",
    "Speech Synthesis": "语音合成"
}

# 等级映射：英文 → 中文
LEVEL_MAP = {
    "Beginner": "初级",
    "Intermediate": "中级",
    "Advanced": "高级"
}


# 注册静态文件服务
app.static_folder = "/data/huangtianle/Ai-Voice-Platform/static"
# ====== WebSocket 实时 ASR 推流新增 ======
ASR_LOG_Q = queue.Queue()          # 留给前端轮询备用，可忽略
STREAM_ASR_THREAD = None           # 读 stdout 线程句柄

# JWT 配置
SECRET_KEY = '4079ba7c3c6f7edc7e821316c6c086a1a653fe376ca4c0d0cfa229792e4169a2'  # 请替换为强密钥（建议使用随机生成）
TOKEN_EXPIRY = 3600 * 24  # Token 有效期 1 小时（秒）

# 创建必要目录并设置权限
for directory in [UPLOAD_DIR, OUTPUT_DIR, CONFIG_DIR, OFFLINE_UPLOAD_DIR, VITS_INFERENCE_DIR, GUIDE_DIR]:
    os.makedirs(directory, exist_ok=True)
    try:
        os.chmod(directory, 0o766)
    except PermissionError as e:
        print(f"Warning: Unable to set permissions for {directory}: {str(e)}")

# 初始化日志文件
log_file = os.path.join(BASE_DIR, "synthesis_log.txt")
os.makedirs(os.path.dirname(log_file), exist_ok=True)
logging.basicConfig(
    filename=log_file,
    level=logging.DEBUG,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
# 把 Flask 默认日志也打到同一个文件
logging.getLogger('werkzeug').setLevel(logging.DEBUG)
logging.getLogger('werkzeug').addHandler(logging.FileHandler(log_file))
logger = logging.getLogger(__name__)

# 替换所有 print 为 logger
for directory in [UPLOAD_DIR, OUTPUT_DIR, CONFIG_DIR, OFFLINE_UPLOAD_DIR, VITS_INFERENCE_DIR]:
    os.makedirs(directory, exist_ok=True)
    try:
        os.chmod(directory, 0o766)
    except PermissionError as e:
        logger.warning(f"Unable to set permissions for {directory}: {str(e)}")

logger.info("Server started")

# 初始化用户 JSON 文件
# if not os.path.exists(USERS_FILE):
#     with open(USERS_FILE, 'w', encoding='utf-8') as f:
#         json.dump([{"ip": "127.0.0.1", "role": "admin"}], f)
#     os.chmod(USERS_FILE, 0o666)

# 数据库连接
# def get_db():
#     db_path = os.path.join(BASE_DIR, 'projects.db')
#     logger.debug(f"Opening database at: {db_path}")
#     if not os.path.exists(db_path):
#         logger.error(f"Database file {db_path} does not exist")
#         raise sqlite3.OperationalError(f"Database file {db_path} does not exist")
#     try:
#         conn = sqlite3.connect(db_path, check_same_thread=False)
#         conn.row_factory = sqlite3.Row
#         # ★★★ 每次新连接都强制打开外键 ★★★
#         conn.execute('PRAGMA foreign_keys = ON')
#         # 顺手再验证一次，方便排错
#         fk_status = conn.execute('PRAGMA foreign_keys').fetchone()[0]
#         logger.info(f"Foreign keys status for this connection: {fk_status}")
#         return conn
#     except sqlite3.OperationalError as e:
#         logger.error(f"Failed to connect to database {db_path}: {str(e)}")
#         raise

# # 初始化用户表
# def init_db():
#     with get_db() as conn:
#         c = conn.cursor()
#         # 启用外键约束
#         c.execute('PRAGMA foreign_keys = ON')
        
#         # 创建 users 表
#         c.execute('''
#             CREATE TABLE IF NOT EXISTS users (
#                 username TEXT PRIMARY KEY,
#                 password TEXT NOT NULL,
#                 role TEXT NOT NULL
#             )
#         ''')
#         # 创建 projects 表
#         c.execute('''
#             CREATE TABLE IF NOT EXISTS projects (
#                 id INTEGER PRIMARY KEY AUTOINCREMENT,
#                 name TEXT NOT NULL,
#                 type TEXT NOT NULL,
#                 level TEXT NOT NULL,
#                 created_at TEXT NOT NULL,
#                 guide_path TEXT
#             )
#         ''')
#         # 创建 distributed_projects 表
#         c.execute('''
#             CREATE TABLE IF NOT EXISTS distributed_projects (
#                 project_id INTEGER,
#                 student_ip TEXT,
#                 PRIMARY KEY (project_id, student_ip),
#                 FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE
#             )
#         ''')
#         # 创建 student_ips 表
#         c.execute('''
#             CREATE TABLE IF NOT EXISTS student_ips (
#                 ip TEXT PRIMARY KEY
#             )
#         ''')
#         # 插入默认用户
#         default_users = [
#             ('admin', '$2b$12$jOFP6O.zz/AK6x/hppKyWeGd4Mams/Yarl6QRPR/3/7fLZ9F89Req', 'admin'),
#             ('student1', '$2b$12$8PkJ8o7mgZo1ZIHTh60C6uE99fiFTeyujLn5MmvrKOKbbp0WHnA0m', 'student')
#         ]
#         for username, password, role in default_users:
#             c.execute('INSERT OR IGNORE INTO users (username, password, role) VALUES (?, ?, ?)',
#                       (username, password, role))
#         conn.commit()
#         # 验证表创建
#         c.execute("SELECT name FROM sqlite_master WHERE type='table'")
#         tables = [row['name'] for row in c.fetchall()]
#         logger.info(f"Database initialized with tables: {tables}")

# 注册 Ubuntu 自带中文字体
try:
    # 优先使用 WenQuanYi Zen Hei
    font_path = '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc'
    if os.path.exists(font_path):
        pdfmetrics.registerFont(TTFont('WenQuanYiZenHei', font_path))
    else:
        # 回退到 Noto Sans CJK
        font_path = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
        pdfmetrics.registerFont(TTFont('NotoSansCJK', font_path))
except Exception as e:
    print(f"Error registering font: {str(e)}")

training_process = None
asr_training_process = None
log_queue = queue.Queue()
asr_log_queue = queue.Queue()
training_thread = None
asr_training_thread = None
process_killed = False
asr_process_killed = False

def allowed_file(filename, allowed_extensions=ALLOWED_EXTENSIONS):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

def extract_zip(zip_path, extract_to):
    logger.debug(f"Extracting ZIP: {zip_path} to {extract_to}")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.testzip()  # 测试 ZIP 文件完整性
            zip_ref.extractall(extract_to)
        items = os.listdir(extract_to)
        if len(items) == 1 and os.path.isdir(os.path.join(extract_to, items[0])):
            inner_dir = os.path.join(extract_to, items[0])
            for item in os.listdir(inner_dir):
                shutil.move(os.path.join(inner_dir, item), extract_to)
            shutil.rmtree(inner_dir)
        logger.debug(f"Extracted ZIP to {extract_to}: {items}")
        return True
    except zipfile.BadZipFile as e:
        logger.error(f"Bad ZIP file: {str(e)}")
        return f"Bad ZIP file: {str(e)}"
    except Exception as e:
        logger.error(f"Extract error: {str(e)}")
        return str(e)

###添加 JWT 装饰器
def require_auth(role=None):
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            auth_header = request.headers.get('Authorization')
            if not auth_header or not auth_header.startswith('Bearer '):
                return jsonify({'error': '缺少令牌'}), 401
            token = auth_header.split(' ')[1]
            try:
                payload = jwt.decode(token, SECRET_KEY, algorithms=['HS256'])
                request.user = payload
                if role and payload.get('role') != role:
                    return jsonify({'error': '无权限执行此操作'}), 403
                return f(*args, **kwargs)
            except jwt.ExpiredSignatureError:
                return jsonify({'error': '令牌已过期'}), 401
            except jwt.InvalidTokenError:
                return jsonify({'error': '无效令牌'}), 401
        return decorated_function
    return decorator

def get_local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except:
        return "127.0.0.1"

# ====== 读取 speed_test.py stdout 并推送到 WebSocket ======
def _asr_reader():
    global STREAM_ASR_PROC
    while True:
        if STREAM_ASR_PROC is None or STREAM_ASR_PROC.poll() is not None:
            time.sleep(0.1)
            continue

        line = STREAM_ASR_PROC.stdout.readline()
        if not line:
            time.sleep(0.1)
            continue

        line = line.strip()
        if not line:
            continue

        logger.debug(f"[DEBUG _asr_reader] raw line: {repr(line)}")

        # 提取 [start - end] text 格式
        import re
        match = re.search(r'\[.*?\]\s*(.+)', line)
        if match:
            text = match.group(1).strip()
            if text:
                logger.debug(f"[DEBUG socketio] emit asr_text: {text}")
                socketio.emit('asr_text', {'text': text})
        else:
            # 其他日志也发过去（可选）
            pass

# ===== 协程安全版：使用 socketio.start_background_task 启动 =====
def asr_reader_task():
    """在 eventlet 协程中读取子进程 stdout 并推送 asr_text 事件"""
    import re
    while True:
        # 进程已结束或未启动
        if not STREAM_ASR_PROC or STREAM_ASR_PROC.poll() is not None:
            time.sleep(0.1)
            continue

        line = STREAM_ASR_PROC.stdout.readline()
        if not line:
            time.sleep(0.1)
            continue

        line = line.strip()
        if not line:
            continue

        logger.debug(f"[ASR_READER] raw: {line}")

        # 提取 [时间] 文本 格式
        match = re.search(r'\[.*?\]\s*(.+)', line)
        if match:
            text = match.group(1).strip()
            if text:
                logger.debug(f"[EMIT] asr_text: {text}")
                socketio.emit('asr_text', {'text': text})
        # 可选：过滤启动日志
        elif any(phrase in line for phrase in ['正在加载模型', '实时语音识别启动', '请说']):
            pass


SECRET_KEY = '4079ba7c3c6f7edc7e821316c6c086a1a653fe376ca4c0d0cfa229792e4169a2'
TOKEN_EXPIRY = 3600 * 24                       # 1 天
FIXED_PASSWORD_MD5 = 'e10adc3949ba59abbe56e057f20f883e'   # 123456 的 MD5
USER_ROOT = Path.home() / 'Ai-Voice-Platform' / 'User'      # ~/Ai-Voice-Platform/User

# 确保根目录存在
USER_ROOT.mkdir(parents=True, exist_ok=True)

###登录端点
@app.route('/api/login', methods=['POST'])
def login():
    """
    无需数据库的登录接口
    账号：5-10 位英文字母+数字
    密码：123456（固定）
    """
    data = request.get_json()
    if not data:
        return jsonify({'error': '请求体非 JSON'}), 400

    username = str(data.get('username', '')).strip()
    password = str(data.get('password', '')).strip()

    # 1. 账号格式校验
    # 特例：管理员账号 admin/admin
    if username == 'admin' and password == 'admin':
        pass  # 跳过格式和密码校验
    else:
        # 普通用户校验
        if not re.fullmatch(r'^[A-Za-z0-9]{5,10}$', username):
            return jsonify({'error': '账号仅限 5-10 位英文字母与数字组合'}), 400

        if hashlib.md5(password.encode('utf-8')).hexdigest() != FIXED_PASSWORD_MD5:
            return jsonify({'error': '密码错误'}), 401

    # 3. 创建用户目录
    user_dir = USER_ROOT / username
    user_dir.mkdir(exist_ok=True)

    # 4. 生成 JWT
    token = jwt.encode(
        {
            'username': username,
            'role': 'admin' if username == 'admin' else 'student',  # ✅ 正确设置角色
            'exp': datetime.utcnow() + timedelta(seconds=TOKEN_EXPIRY)
        },
        SECRET_KEY,
        algorithm='HS256'
    )

    # 5. 返回约定格式
    return jsonify({
        'code': 200,
        'msg': 'success',
        'userList': {
            'token': token,
            'role': 2,                   # 与旧前端保持一致
            'user': username
        }
    }), 200

# ① 获取模型列表
@app.route('/api/stream_asr/models', methods=['GET'])
def stream_asr_models():
    asr_model_dir = Path(__file__).with_name('models') / 'ASR_models'
    if not asr_model_dir.exists():
        return jsonify({'models': []})
    ms = [d.name for d in asr_model_dir.iterdir() if d.is_dir()]
    return jsonify({'models': ms})

# ② 启动识别（后台挂起 speed_test.py）
@app.route('/api/stream_asr/start', methods=['POST'])
@require_auth()
def stream_asr_start():
    global STREAM_ASR_PROC

    try:
        data = request.get_json(force=True)
        model = data.get('model')
        if not model:
            return jsonify({'error': '缺少 model 参数'}), 400

        with STREAM_ASR_LOCK:
            if STREAM_ASR_PROC and STREAM_ASR_PROC.poll() is None:
                return jsonify({'error': '已有识别任务在运行'}), 400

            # ===== 启动 speed_test.py 子进程 =====
            cli_py = Path(BASE_DIR) / 'speed_test.py'
            python_exe = sys.executable  # 当前 Python 环境

            # 动态设置模型路径（如果你支持多模型）
            env = os.environ.copy()
            env['MODEL_PATH'] = str(Path(MODEL_DIR) / model)

            STREAM_ASR_PROC = subprocess.Popen(
                [python_exe, '-u', str(cli_py)],
                cwd=BASE_DIR,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=env
            )
            logger.info(f'[STREAM-ASR] 子进程启动，PID={STREAM_ASR_PROC.pid}')

            # 关键：使用 eventlet 协程安全启动读取任务
            socketio.start_background_task(asr_reader_task)

        return jsonify({'message': '实时识别已启动'})

    except Exception as e:
        logger.error(f"[STREAM-ASR] 启动失败: {str(e)}")
        return jsonify({'error': str(e)}), 500

# ③ 停止识别
@app.route('/api/stream_asr/stop', methods=['POST'])
@require_auth()
def stream_asr_stop():
    global STREAM_ASR_PROC
    with STREAM_ASR_LOCK:
        if STREAM_ASR_PROC and STREAM_ASR_PROC.poll() is None:
            STREAM_ASR_PROC.terminate()
            try:
                STREAM_ASR_PROC.wait(timeout=3)
            except subprocess.TimeoutExpired:
                STREAM_ASR_PROC.kill()
            logger.info("[STREAM-ASR] 子进程已终止")
        STREAM_ASR_PROC = None
    return jsonify({'message': '已停止'})

# ④ SSE 推送识别结果（speed_test.py 每输出一行就推）
@app.route('/api/stream_asr/listen', methods=['GET'])
def stream_asr_listen():
    def generate():
        proc = STREAM_ASR_PROC
        if not proc or proc.poll() is not None:
            yield f"data: {json.dumps({'text': ''})}\n\n"
            return
        while True:
            line = proc.stdout.readline()
            print(f"[DEBUG] raw line from speed_test: {repr(line)}")
            if not line:
                time.sleep(0.1)
                continue
            # speed_test.py 的 print 格式： "[ 0.00s - 1.20s] 你好语音识别"
            if ']' in line:
                text = line.split(']', 1)[-1].strip()
                yield f"data: {json.dumps({'text': text})}\n\n"
    return Response(generate(), mimetype="text/event-stream")




###获取模型
@app.route('/api/models', methods=['GET'])
def get_models():
    try:
        models = [f for f in os.listdir(MODEL_DIR) if os.path.isdir(os.path.join(MODEL_DIR, f))]
        return jsonify({"models": models})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/upload', methods=['POST'])
def upload_dataset():
    if 'file' not in request.files:
        return jsonify({"error": "没有上传文件"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "未选择文件"}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        upload_path = os.path.join(UPLOAD_DIR, filename)
        file.save(upload_path)
        extract_dir = os.path.join(UPLOAD_DIR, os.path.splitext(filename)[0])
        os.makedirs(extract_dir, exist_ok=True)
        result = extract_zip(upload_path, extract_dir)
        if result is True:
            return jsonify({"message": "数据集上传并解压成功", "dataset_path": extract_dir})
        else:
            return jsonify({"error": f"解压失败: {result}"}), 500
    return jsonify({"error": "文件格式不支持"}), 400

@app.route('/api/upload_vits_dataset', methods=['POST'])
@require_auth()
def upload_vits_dataset():
    logger.debug("Received upload_vits_dataset request")
    if 'file' not in request.files:
        logger.error("No file part in request")
        return jsonify({'error': '没有上传文件'}), 400
    file = request.files['file']
    if file.filename == '':
        logger.error("No file selected")
        return jsonify({'error': '未选择文件'}), 400
    if file and allowed_file(file.filename, ALLOWED_EXTENSIONS):
        filename = secure_filename(file.filename)
        name = os.path.splitext(filename)[0]
        upload_path = os.path.join(UPLOAD_DIR, filename)
        logger.debug(f"Saving file to {upload_path}")
        try:
            file.save(upload_path)
            os.chmod(upload_path, 0o666)
        except Exception as e:
            logger.error(f"Error saving file {upload_path}: {str(e)}")
            return jsonify({'error': f'文件保存失败: {str(e)}'}), 500

        tmp_extract = os.path.join(UPLOAD_DIR, f"{name}_tmp")
        os.makedirs(tmp_extract, exist_ok=True)
        try:
            os.chmod(tmp_extract, 0o766)
        except PermissionError as e:
            logger.warning(f"Unable to set permissions for {tmp_extract}: {str(e)}")

        logger.debug(f"Extracting ZIP to {tmp_extract}")
        result = extract_zip(upload_path, tmp_extract)
        if result is not True:
            shutil.rmtree(tmp_extract, ignore_errors=True)
            logger.error(f"Extraction failed: {result}")
            return jsonify({'error': f'解压失败: {result}'}), 500

        target_base = VITS_BASE
        os.makedirs(target_base, exist_ok=True)
        try:
            os.chmod(target_base, 0o766)
        except PermissionError as e:
            logger.warning(f"Unable to set permissions for {target_base}: {str(e)}")

        logger.debug(f"Clearing old contents in {target_base}")
        for entry in os.listdir(target_base):
            path = os.path.join(target_base, entry)
            try:
                if os.path.isdir(path):
                    shutil.rmtree(path)
                else:
                    os.remove(path)
            except Exception as e:
                logger.warning(f"无法删除 {path}: {e}")

        target_dir = os.path.join(target_base, name)
        if os.path.exists(target_dir):
            shutil.rmtree(target_dir, ignore_errors=True)
        os.makedirs(target_dir, exist_ok=True)
        try:
            os.chmod(target_dir, 0o766)
        except PermissionError as e:
            logger.warning(f"Unable to set permissions for {target_dir}: {str(e)}")

        logger.debug(f"Moving files from {tmp_extract} to {target_dir}")
        for item in os.listdir(tmp_extract):
            s = os.path.join(tmp_extract, item)
            d = os.path.join(target_dir, item)
            try:
                shutil.move(s, d)
            except Exception as e:
                logger.error(f"Error moving {s} to {d}: {str(e)}")
                shutil.rmtree(tmp_extract, ignore_errors=True)
                return jsonify({'error': f'文件移动失败: {str(e)}'}), 500
        shutil.rmtree(tmp_extract, ignore_errors=True)

        subfolders = sorted([d for d in os.listdir(target_dir) if os.path.isdir(os.path.join(target_dir, d))])
        debug_info = {
            'top_level': sorted(os.listdir(target_dir)),
            'subfolders': {}
        }

        logger.debug(f"Processing subfolders: {subfolders}")
        train_lines = []
        val_lines = []

        for idx, sub in enumerate(subfolders):
            sub_path = os.path.join(target_dir, sub)
            wavs = sorted([f for f in os.listdir(sub_path) if f.lower().endswith('.wav')])
            txts = sorted([f for f in os.listdir(sub_path) if f.lower().endswith(('.txt', '.lab', '.text'))])
            debug_info['subfolders'][sub] = {'wav_count': len(wavs), 'txt_count': len(txts)}

            pairs = []
            missing_txt = []
            for wav in wavs:
                base = os.path.splitext(wav)[0]
                txt_path = os.path.join(sub_path, base + '.txt')
                if not os.path.exists(txt_path):
                    for ext in ['.lab', '.text']:
                        if os.path.exists(os.path.join(sub_path, base + ext)):
                            txt_path = os.path.join(sub_path, base + ext)
                            break
                if not os.path.exists(txt_path):
                    missing_txt.append(wav)
                    continue

                try:
                    with open(txt_path, 'r', encoding='utf-8') as f:
                        lines = [ln.strip() for ln in f.readlines() if ln.strip()]
                    text_line = lines[1] if len(lines) > 1 else (lines[0] if lines else '')
                except Exception as e:
                    logger.error(f"Error reading {txt_path}: {str(e)}")
                    text_line = ''
                rel_path = f"./dataset/{name}/{sub}/{wav}"
                pairs.append((rel_path, str(idx), text_line))

            debug_info['subfolders'][sub]['missing_txt'] = missing_txt

            n = len(pairs)
            if n == 0:
                continue
            split = int(n * 0.9)
            if split < 1:
                split = n - 1 if n > 1 else 1
            for i, item in enumerate(pairs):
                line = '|'.join(item)
                if i < split:
                    train_lines.append(line)
                else:
                    val_lines.append(line)

        if not train_lines and not val_lines:
            logger.error("No valid wav-txt pairs found")
            return jsonify({'error': '未找到有效的 wav-txt 配对样本', 'debug': debug_info}), 400

        train_file = os.path.join(target_base, f"{name}_train.txt")
        val_file = os.path.join(target_base, f"{name}_val.txt")
        logger.debug(f"Writing train file: {train_file}")
        try:
            with open(train_file, 'w', encoding='utf-8') as f:
                for ln in train_lines:
                    f.write(ln + '\n')
            with open(val_file, 'w', encoding='utf-8') as f:
                for ln in val_lines:
                    f.write(ln + '\n')
        except Exception as e:
            logger.error(f"Error writing train/val files: {str(e)}")
            return jsonify({'error': f'写入 train/val 文件失败: {str(e)}', 'debug': debug_info}), 500

        def generate_symbols(train_file_path, val_file_path):
            symbols = set()
            for file_path in [train_file_path, val_file_path]:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        for line in f:
                            parts = line.strip().split('|')
                            if len(parts) < 3:
                                continue
                            text = parts[2].strip()
                            if not text:
                                continue
                            tokens = text.split()
                            for t in tokens:
                                if t:
                                    symbols.add(t)
                except Exception as e:
                    logger.error(f"Error reading {file_path} for symbols: {str(e)}")
                    continue
            return sorted(list(symbols))

        symbols = generate_symbols(train_file, val_file)
        symbols_file = os.path.join(target_base, f"{name}_symbols.txt")
        logger.debug(f"Writing symbols file: {symbols_file}")
        try:
            with open(symbols_file, 'w', encoding='utf-8') as f:
                f.write('[\n')
                for i, s in enumerate(symbols):
                    comma = ',' if i != len(symbols) - 1 else ''
                    f.write(f'  "{s}"{comma}\n')
                f.write(']\n')
        except Exception as e:
            logger.error(f"Error writing symbols file: {str(e)}")
            return jsonify({'error': f'写入 symbols 文件失败: {str(e)}', 'debug': debug_info}), 500

        logger.info(f"Dataset uploaded: {target_dir}")
        return jsonify({
            'message': '上传并处理成功',
            'dataset_name': name,
            'dataset_dir': os.path.abspath(target_dir),
            'train_file': os.path.abspath(train_file),
            'val_file': os.path.abspath(val_file),
            'symbols_file': os.path.abspath(symbols_file),
            'debug': debug_info
        })
    logger.error(f"File type not allowed: {file.filename}")
    return jsonify({'error': '文件格式不支持，需为 zip'}), 400

# ============================== ASR 微调专用上传接口 ==============================
@app.route('/api/asr_finetune_upload', methods=['POST'])
@require_auth()
def asr_finetune_upload():
    """
    前端调用：/api/asr_finetune_upload?folder=19
    上传 zip → 解压到 /data/huangtianle/Ai-Voice-Platform/User/admin/<folder>/dataset/
    """
    folder_name = request.args.get('folder')
    if not folder_name:
        return jsonify({"error": "URL 中缺少 folder 参数"}), 400

    safe_folder = secure_filename(folder_name)
    if safe_folder != folder_name:
        return jsonify({"error": "文件夹名称包含非法字符"}), 400

    target_root = USER_DATA_ROOT / "admin" / safe_folder / "dataset"
    target_root.mkdir(parents=True, exist_ok=True)

    if 'file' not in request.files:
        return jsonify({"error": "没有上传文件"}), 400
    file = request.files['file']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({"error": "文件必须是 .zip"}), 400

    tmp_zip = Path(UPLOAD_DIR) / f"asr_finetune_tmp_{uuid.uuid4().hex}.zip"
    file.save(str(tmp_zip))

    try:
        result = extract_zip(str(tmp_zip), str(target_root))
        if result is not True:
            return jsonify({"error": f"解压失败: {result}"}), 500
    finally:
        if tmp_zip.exists():
            tmp_zip.unlink()

    return jsonify({
        "message": "数据集上传并解压成功",
        "dataset_path": str(target_root)
    })
# ==============================================================================

@app.route('/api/confirm_vits_params', methods=['POST'])
def confirm_vits_params():
    data = request.get_json()
    if not data:
        return jsonify({'error': '缺少参数数据'}), 400

    training_files = data.get('data', {}).get('training_files')
    if not training_files:
        return jsonify({'error': '缺少 training_files'}), 400

    try:
        dataset_name = os.path.splitext(os.path.basename(training_files))[0]
        dataset_name = dataset_name.replace('_train', '').replace('_val', '')
        if not dataset_name:
            return jsonify({'error': '无法从 training_files 提取有效的数据集名称'}), 400
    except Exception as e:
        return jsonify({'error': f'无法提取数据集名称: {str(e)}'}), 400

    try:
        config_path = os.path.join(BASE_DIR, 'VITS-fast-fine-tuning', 'configs', 'modified_finetune_speaker.json')
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)

        config['train'].update(data['train'])
        config['data']['training_files'] = data['data']['training_files']
        config['data']['validation_files'] = data['data']['validation_files']
        config['data']['text_cleaners'] = data['data']['text_cleaners']
        config['data']['n_speakers'] = data['data']['n_speakers']
        config['model']['gin_channels'] = data['model']['gin_channels']
        config['model']['speakers'] = data['model']['speakers']
        symbols_file = data.get('symbols_file')
        if not os.path.exists(symbols_file):
            return jsonify({'error': f'符号文件 {symbols_file} 不存在'}), 400
        with open(symbols_file, 'r', encoding='utf-8') as f:
            config['symbols'] = json.load(f)
        config['preserved'] = data.get('preserved', 2)

        config_filename = f"{dataset_name}_finetune_speaker.json"
        config_path = os.path.join(CONFIG_DIR, config_filename)
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)

        text_cleaners = data['data']['text_cleaners'][0].replace('_cleaners', '')
        symbols = config['symbols']
        pinyin_func = 'pc.characters_to_jyutping(text)' if text_cleaners.lower() == 'yueyu' else 'pc.pinyin(text, style=pc.Style.TONE)'
        pinyin_var = 'jyutping_list' if text_cleaners.lower() == 'yueyu' else 'pinyin_list'
        cleaners_content = f"""# BEGIN_CUSTOM_CLEANERS
{text_cleaners.upper()}_SYMBOLS = set({json.dumps(symbols, ensure_ascii=False)})

def {text_cleaners}_cleaners(text: str) -> str:
    import re
    import pypinyin as pc
    text = re.sub(r'[^\\u4e00-\\u9fff]', ' ', text)
    {pinyin_var} = {pinyin_func}

    phones = []
    for py in {pinyin_var}:
        py = py[1] if isinstance(py, tuple) else py[0]
        if py and py.lower() in {text_cleaners.upper()}_SYMBOLS:
            phones.append(py.lower())
        else:
            phones.append('<unk>')
    
    return ' '.join(phones).strip() or '<unk>'
# END_OF_CUSTOM_CLEANERS
"""

        original_content = ""
        if os.path.exists(CLEANERS_FILE):
            with open(CLEANERS_FILE, 'r', encoding='utf-8') as f:
                original_content = f.read()

        begin_marker = "# BEGIN_CUSTOM_CLEANERS"
        end_marker = "# END_OF_CUSTOM_CLEANERS"
        new_content = original_content

        if begin_marker in original_content and end_marker in original_content:
            start_idx = original_content.index(begin_marker)
            end_idx = original_content.index(end_marker) + len(end_marker)
            new_content = original_content[:start_idx] + cleaners_content + original_content[end_idx:]
        else:
            new_content = original_content.rstrip() + "\n\n" + cleaners_content

        with open(CLEANERS_FILE, 'w', encoding='utf-8') as f:
            f.write(new_content)

        return jsonify({
            'message': '参数确认成功，cleaners.py 已更新',
            'dataset_name': dataset_name,
            'config_path': os.path.abspath(config_path)
        })
    except Exception as e:
        return jsonify({'error': f'参数处理失败: {str(e)}'}), 500

@app.route('/api/train_vits', methods=['POST'])
@require_auth()
def train_vits_model():
    global training_process, process_killed, training_thread
    data = request.get_json()
    model_save_path = data.get('model_save_path')
    config_path = data.get('config_path')
    preserved = data.get('preserved', 2)

    if not model_save_path:
        return jsonify({"error": "缺少 model_save_path"}), 400
    if not config_path:
        return jsonify({"error": "缺少 config_path"}), 400
    if not os.path.exists(config_path):
        return jsonify({"error": "配置文件不存在，请先确认参数"}), 400

    conda_env_path = os.path.join(os.path.expanduser("~"), "anaconda3", "envs", "VITS_train_env")
    python_executable = os.path.join(conda_env_path, "bin", "python")

    if not os.path.exists(python_executable):
        error_msg = f"Python executable not found: {python_executable}"
        with open(os.path.join(BASE_DIR, "train_log.txt"), 'a', encoding='utf-8') as f:
            f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] ERROR: {error_msg}\n")
        return jsonify({"error": error_msg}), 500

    cmd = [
        python_executable,
        os.path.join(BASE_DIR, "VITS-fast-fine-tuning", "finetune_speaker_v3.py"),
        "-m", model_save_path,
        "--drop_speaker_embed", "True",
        "-c", config_path,
        "--preserved", str(preserved)
    ]

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PATH"] = f"{os.path.join(conda_env_path, 'bin')}:{env['PATH']}"
    env["LD_LIBRARY_PATH"] = f"{os.path.join(conda_env_path, 'lib')}:/usr/lib/x86_64-linux-gnu:{env.get('LD_LIBRARY_PATH', '')}"
    env["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0")

    log_file = os.path.join(BASE_DIR, "train_log.txt")
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Executing command: {' '.join(cmd)}\n")
        f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Environment PATH: {env['PATH']}\n")
        f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Environment LD_LIBRARY_PATH: {env['LD_LIBRARY_PATH']}\n")
        f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Environment CUDA_VISIBLE_DEVICES: {env['CUDA_VISIBLE_DEVICES']}\n")
        sys.stdout.flush()

    def run_training():
        global training_process, process_killed
        try:
            test_cmd = [python_executable, "-c", "import sys, torch, torio; print('Test: Python version: %s, CUDA: %s, FFmpeg: %s' % (sys.version, torch.cuda.is_available(), torio._extension._FFMPEG_EXT_LOADED)); sys.stdout.flush()"]
            test_process = subprocess.run(
                test_cmd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=10
            )
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Test command STDOUT: {test_process.stdout}\n")
                f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Test command STDERR: {test_process.stderr}\n")
                if test_process.returncode != 0:
                    f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Test command failed with exit code {test_process.returncode}\n")
            log_queue.put(f"Test command output: {test_process.stdout.strip()}")
            if test_process.stderr:
                log_queue.put(f"Test command error: {test_process.stderr.strip()}")

            training_process = subprocess.Popen(
                cmd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                universal_newlines=True,
                cwd=os.path.join(BASE_DIR, "VITS-fast-fine-tuning")
            )
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Training process started with PID: {training_process.pid}\n")
                sys.stdout.flush()

            from select import select
            while training_process.poll() is None:
                rlist, _, _ = select([training_process.stdout, training_process.stderr], [], [], 0.1)
                for pipe in rlist:
                    line = pipe.readline().strip()
                    if line:
                        log_queue.put(line)
                        with open(log_file, 'a', encoding='utf-8') as f:
                            f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {'STDOUT' if pipe == training_process.stdout else 'STDERR'}: {line}\n")
                sys.stdout.flush()
            for line in training_process.stdout:
                if line.strip():
                    log_queue.put(line.strip())
                    with open(log_file, 'a', encoding='utf-8') as f:
                        f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] STDOUT: {line.strip()}\n")
            for line in training_process.stderr:
                if line.strip():
                    log_queue.put(line.strip())
                    with open(log_file, 'a', encoding='utf-8') as f:
                        f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] STDERR: {line.strip()}\n")
            training_process.wait()
            return_code = training_process.returncode
            if return_code == 0 and not process_killed:
                success_msg = f"Training completed, model saved to {model_save_path}"
                log_queue.put(success_msg)
                with open(log_file, 'a', encoding='utf-8') as f:
                    f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {success_msg}\n")
            else:
                error_msg = f"Training failed with exit code {return_code}"
                log_queue.put(error_msg)
                with open(log_file, 'a', encoding='utf-8') as f:
                    f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {error_msg}\n")
        except Exception as e:
            error_msg = f"Training failed: {str(e)}"
            log_queue.put(error_msg)
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {error_msg}\n")
            sys.stdout.flush()

    if training_process and training_process.poll() is None:
        process_killed = True
        training_process.terminate()
        try:
            training_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            training_process.kill()
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Previous training process terminated\n")
        training_process = None

    training_thread = Thread(target=run_training)
    training_thread.start()
    return jsonify({"message": "训练已开始"})

def stream_logs():
    log_file = os.path.join(BASE_DIR, "train_log.txt")
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] SSE connection established\n")
    last_heartbeat = time.time()
    while True:
        try:
            log = log_queue.get_nowait()
            yield f"data: {json.dumps({'message': log})}\n\n"
            last_heartbeat = time.time()
        except queue.Empty:
            if process_killed:
                yield f"data: {json.dumps({'message': '训练已停止'})}\n\n"
                break
            # 发送心跳消息防止连接超时
            if time.time() - last_heartbeat > 10:
                yield f"data: {json.dumps({'message': 'Heartbeat: Connection alive'})}\n\n"
                last_heartbeat = time.time()
            time.sleep(0.1)

@app.route('/api/stop', methods=['POST'])
def stop_training():
    global training_process, process_killed
    if training_process and training_process.poll() is None:
        process_killed = True
        training_process.send_signal(signal.SIGTERM)
        try:
            training_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            training_process.kill()
        training_process = None
        log_file = os.path.join(BASE_DIR, "train_log.txt")
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] VITS training stopped by user\n")
        # 清空 VITS 日志队列
        while not log_queue.empty():
            log_queue.get()
        return jsonify({"message": "VITS 训练已停止"})
    return jsonify({"error": "没有正在进行的 VITS 训练任务"}), 400

@app.route('/api/train', methods=['GET'])
def stream_training_logs():
    return Response(stream_logs(), mimetype='text/event-stream')

@app.route('/api/outputs/<path:filename>', methods=['GET'])
def download_output():
    try:
        return send_from_directory(OUTPUT_DIR, filename, as_attachment=True)
    except Exception as e:
        return jsonify({"error": str(e)}), 404

@app.route('/api/asr_models', methods=['GET'])
def get_asr_models():
    asr_model_dir = './models/ASR_models'
    try:
        models = [f for f in os.listdir(asr_model_dir) if os.path.isdir(os.path.join(asr_model_dir, f))]
        return jsonify({'models': models})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/upload_audio', methods=['POST'])
def upload_audio():
    audio_dir = os.path.join(UPLOAD_DIR, 'audio')
    os.makedirs(audio_dir, exist_ok=True)
    if 'file' not in request.files:
        return jsonify({'error': '没有上传文件'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': '未选择文件'}), 400
    filename = secure_filename(file.filename)
    save_path = os.path.join(audio_dir, filename)
    file.save(save_path)
    return jsonify({'message': '音频上传成功', 'audio_path': save_path})

@app.route('/api/recognize', methods=['POST'])
def recognize_audio():
    data = request.get_json()
    model_name = data.get('model_name')
    audio_path = data.get('audio_path')
    if not model_name or not audio_path:
        return jsonify({'error': '缺少 model_name 或 audio_path'}), 400
    model_dir = os.path.join('./models/ASR_models', model_name)
    test_script = os.path.abspath('test.py')
    conda_activate = 'source ~/anaconda3/bin/activate ASR_train_env'
    cmd = f"{conda_activate} && python {test_script} --model_path '{model_dir}' --audio_path '{audio_path}'"
    try:
        result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, executable='/bin/bash', timeout=120)
        output = result.stdout + '\n' + result.stderr
        lines = output.splitlines()
        transcription = ''
        for line in lines:
            if line.startswith('Transcription:'):
                transcription = line.replace('Transcription:', '').strip()
                break
        return jsonify({'transcription': transcription, 'raw_output': output})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# 只给 ASR 微调用，保存到 Uploads/asr 目录


@app.route('/api/asr_train_logs', methods=['GET'])
def stream_asr_training_logs():
    print("【DEBUG】接收到的 data =", data)
    return Response(stream_asr_logs(), mimetype='text/event-stream')

def stream_logs():
    log_file = os.path.join(BASE_DIR, "train_log.txt")
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] VITS SSE connection established\n")
    last_heartbeat = time.time()
    while True:
        try:
            log = log_queue.get_nowait()
            yield f"data: {json.dumps({'message': log})}\n\n"
            last_heartbeat = time.time()
        except queue.Empty:
            if process_killed:
                yield f"data: {json.dumps({'message': 'VITS 训练已停止'})}\n\n"
                break
            if time.time() - last_heartbeat > 10:
                yield f"data: {json.dumps({'message': 'Heartbeat: Connection alive'})}\n\n"
                last_heartbeat = time.time()
            time.sleep(0.1)

def stream_asr_logs():
    log_file = os.path.join(BASE_DIR, "train_log.txt")
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] ASR SSE connection established\n")
    last_heartbeat = time.time()
    while True:
        try:
            log = asr_log_queue.get_nowait()
            yield f"data: {json.dumps({'message': log})}\n\n"
            last_heartbeat = time.time()
        except queue.Empty:
            if asr_process_killed:
                yield f"data: {json.dumps({'message': 'ASR 训练已停止'})}\n\n"
                break
            if time.time() - last_heartbeat > 10:
                yield f"data: {json.dumps({'message': 'Heartbeat: Connection alive'})}\n\n"
                last_heartbeat = time.time()
            time.sleep(0.1)

@app.route('/api/asr_train', methods=['POST'])
@require_auth()
def train_asr_model():
    print("【DEBUG】请求头 Authorization =", request.headers.get('Authorization'))
    print("【DEBUG】request.user =", getattr(request, 'user', None))
    data = request.get_json()
    model_name = data.get('model_name')
    folder = data.get('folder')
    training_params = data.get('training_params', {})

    if not model_name or not folder:
        return jsonify({"error": "缺少 model_name 或 folder 参数"}), 400

    username = request.user['username']
    dataset_path = os.path.expanduser(f"~/Ai-Voice-Platform/User/{username}/{folder}/dataset")
    output_dir = os.path.expanduser(f"~/Ai-Voice-Platform/User/{username}/{folder}/ASR_model")

    if not os.path.exists(dataset_path):
        return jsonify({"error": "数据集路径不存在"}), 400

    os.makedirs(output_dir, exist_ok=True)

    # 重置 ASR 训练状态
    asr_process_killed = False
    asr_training_process = None
    asr_training_thread = None
    while not asr_log_queue.empty():
        asr_log_queue.get()

    # 动态生成输出目录
    username = request.headers.get('X-Username')  # 假设用户名通过请求头传递
    if not username:
        return jsonify({"error": "缺少用户名"}), 400

    output_dir = os.path.expanduser(f"~/Ai-Voice-Platform/User/{username}/{folder}/ASR_model")
    os.makedirs(output_dir, exist_ok=True)

    conda_env_path = os.path.join(os.path.expanduser("~"), "anaconda3", "envs", "ASR_train_env")
    python_executable = os.path.join(conda_env_path, "bin", "python")
    
    if not os.path.exists(python_executable):
        error_msg = f"Python executable not found: {python_executable}"
        with open(os.path.join(BASE_DIR, "train_log.txt"), 'a', encoding='utf-8') as f:
            f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] ERROR: {error_msg}\n")
        return jsonify({"error": error_msg}), 500

    cmd = [
        python_executable,
        os.path.join(BASE_DIR, "train.py"),
        "--model_path", os.path.join(MODEL_DIR, model_name),
        "--data_dir", dataset_path,
        "--output_dir", output_dir,
        "--batch_size", str(training_params.get('per_device_train_batch_size', 8)),
        "--gradient_accumulation_steps", str(training_params.get('gradient_accumulation_steps', 4)),
        "--num_train_epochs", str(training_params.get('num_train_epochs', 30)),
        "--learning_rate", str(training_params.get('learning_rate', 1e-5)),
        "--save_steps", str(training_params.get('save_steps', 500)),
        "--logging_steps", str(training_params.get('logging_steps', 100)),
        "--save_total_limit", str(training_params.get('save_total_limit', 2)),
        "--fp16", str(training_params.get('fp16', True)).lower()
    ]

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PATH"] = f"{os.path.join(conda_env_path, 'bin')}:{env['PATH']}"
    env["LD_LIBRARY_PATH"] = f"{os.path.join(conda_env_path, 'lib')}:/usr/lib/x86_64-linux-gnu:{env.get('LD_LIBRARY_PATH', '')}"
    env["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0")

    log_file = os.path.join(BASE_DIR, "train_log.txt")
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Executing ASR training command: {' '.join(cmd)}\n")
        f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Environment PATH: {env['PATH']}\n")
        f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Environment LD_LIBRARY_PATH: {env['LD_LIBRARY_PATH']}\n")
        f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Environment CUDA_VISIBLE_DEVICES: {env['CUDA_VISIBLE_DEVICES']}\n")
        sys.stdout.flush()

    def run_training():
        global asr_training_process, asr_process_killed
        try:
            test_cmd = [python_executable, "-c", "import sys, torch, librosa, transformers; print('Test: Python version: %s, CUDA: %s' % (sys.version, torch.cuda.is_available())); sys.stdout.flush()"]
            test_process = subprocess.run(
                test_cmd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=10
            )
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Test command STDOUT: {test_process.stdout}\n")
                f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Test command STDERR: {test_process.stderr}\n")
                if test_process.returncode != 0:
                    f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Test command failed with exit code {test_process.returncode}\n")
            asr_log_queue.put(f"Test command output: {test_process.stdout.strip()}")
            if test_process.stderr:
                asr_log_queue.put(f"Test command error: {test_process.stderr.strip()}")

            asr_training_process = subprocess.Popen(
                cmd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                universal_newlines=True,
                cwd=BASE_DIR
            )
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] ASR training process started with PID: {asr_training_process.pid}\n")
                sys.stdout.flush()

            from select import select
            while asr_training_process and asr_training_process.poll() is None:
                rlist, _, _ = select([asr_training_process.stdout, asr_training_process.stderr], [], [], 0.1)
                for pipe in rlist:
                    line = pipe.readline().strip()
                    if line:
                        asr_log_queue.put(line)
                        with open(log_file, 'a', encoding='utf-8') as f:
                            f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {'STDOUT' if pipe == asr_training_process.stdout else 'STDERR'}: {line}\n")
                sys.stdout.flush()
            
            # 检查进程是否已结束
            if asr_training_process:
                for line in asr_training_process.stdout:
                    if line.strip():
                        asr_log_queue.put(line.strip())
                        with open(log_file, 'a', encoding='utf-8') as f:
                            f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] STDOUT: {line.strip()}\n")
                for line in asr_training_process.stderr:
                    if line.strip():
                        asr_log_queue.put(line.strip())
                        with open(log_file, 'a', encoding='utf-8') as f:
                            f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] STDERR: {line.strip()}\n")
                return_code = asr_training_process.wait()
                if return_code == 0 and not asr_process_killed:
                    success_msg = f"Training completed, model saved to {output_dir}"
                    asr_log_queue.put(success_msg)
                    with open(log_file, 'a', encoding='utf-8') as f:
                        f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {success_msg}\n")
                else:
                    error_msg = f"Training failed with exit code {return_code}"
                    asr_log_queue.put(error_msg)
                    with open(log_file, 'a', encoding='utf-8') as f:
                        f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {error_msg}\n")
        except Exception as e:
            error_msg = f"Training failed: {str(e)}"
            asr_log_queue.put(error_msg)
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {error_msg}\n")
        finally:
            asr_training_process = None
            asr_training_thread = None
            sys.stdout.flush()

    if asr_training_process and asr_training_process.poll() is None:
        asr_process_killed = True
        asr_training_process.terminate()
        try:
            asr_training_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            asr_training_process.kill()
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Previous ASR training process terminated\n")
        asr_training_process = None
        asr_training_thread = None

    asr_training_thread = Thread(target=run_training)
    asr_training_thread.start()
    return jsonify({"message": "训练已开始", "output_dir": output_dir})

@app.route('/api/asr_stop', methods=['POST'])
def stop_asr_training():
    global asr_training_process, asr_process_killed
    if asr_training_process and asr_training_process.poll() is None:
        asr_process_killed = True
        asr_training_process.send_signal(signal.SIGTERM)
        try:
            asr_training_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            asr_training_process.kill()
        asr_training_process = None
        asr_training_thread = None  # 重置线程
        log_file = os.path.join(BASE_DIR, "train_log.txt")
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] ASR training stopped by user\n")
        # 清空 ASR 日志队列
        while not asr_log_queue.empty():
            asr_log_queue.get()
        return jsonify({"message": "ASR 训练已停止"})
    return jsonify({"error": "没有正在进行的 ASR 训练任务"}), 400

### 模型测试
@app.route('/api/offline_models', methods=['GET'])
def get_offline_models():
    try:
        models = [f for f in os.listdir(OFFLINE_MODEL_DIR) if os.path.isdir(os.path.join(OFFLINE_MODEL_DIR, f))]
        return jsonify({"models": models})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/offline_upload_audio', methods=['POST'])
def offline_upload_audio():
    if 'file' not in request.files:
        return jsonify({"error": "没有上传文件"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "未选择文件"}), 400
    if file and allowed_file(file.filename, ALLOWED_AUDIO_EXTENSIONS):
        filename = secure_filename(file.filename)
        upload_path = os.path.join(OFFLINE_UPLOAD_DIR, f"{uuid.uuid4().hex}_{filename}")
        file.save(upload_path)
        return jsonify({"message": "音频上传成功", "audio_path": upload_path})
    return jsonify({"error": "文件格式不支持，仅支持 .wav"}), 400

@app.route('/api/offline_recognize', methods=['POST'])
def offline_recognize():
    data = request.get_json()
    model_name = data.get('model_name')
    audio_path = data.get('audio_path')
    
    if not model_name:
        return jsonify({"error": "缺少 model_name"}), 400
    if not audio_path:
        return jsonify({"error": "缺少 audio_path"}), 400
    if not os.path.exists(audio_path):
        return jsonify({"error": "音频文件不存在"}), 400
    if not os.path.exists(os.path.join(OFFLINE_MODEL_DIR, model_name)):
        return jsonify({"error": "模型路径不存在"}), 400

    conda_env_path = os.path.join(os.path.expanduser("~"), "anaconda3", "envs", "ASR_train_env")
    python_executable = os.path.join(conda_env_path, "bin", "python")
    
    if not os.path.exists(python_executable):
        return jsonify({"error": f"Python executable not found: {python_executable}"}), 500

    cmd = [
        python_executable,
        os.path.join(BASE_DIR, "test.py"),
        "--model_path", os.path.join(OFFLINE_MODEL_DIR, model_name),
        "--audio_path", audio_path
    ]

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PATH"] = f"{os.path.join(conda_env_path, 'bin')}:{env['PATH']}"
    env["LD_LIBRARY_PATH"] = f"{os.path.join(conda_env_path, 'lib')}:/usr/lib/x86_64-linux-gnu:{env.get('LD_LIBRARY_PATH', '')}"
    env["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0")

    log_file = os.path.join(BASE_DIR, "offline_recognition_log.txt")
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Executing recognition command: {' '.join(cmd)}\n")
        sys.stdout.flush()

    try:
        process = subprocess.run(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=60
        )
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] STDOUT: {process.stdout}\n")
            f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] STDERR: {process.stderr}\n")
            sys.stdout.flush()

        if process.returncode == 0:
            transcription = process.stdout.strip().split("Transcription:")[-1].strip() if "Transcription:" in process.stdout else ""
            return jsonify({"transcription": transcription, "raw_output": process.stdout})
        else:
            error_msg = f"Recognition failed with exit code {process.returncode}: {process.stderr}"
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {error_msg}\n")
            return jsonify({"error": error_msg, "raw_output": process.stderr}), 500
    except subprocess.TimeoutExpired:
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Recognition timed out\n")
        return jsonify({"error": "Recognition timed out"}), 500
    except Exception as e:
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Recognition failed: {str(e)}\n")
        return jsonify({"error": f"Recognition failed: {str(e)}"}), 500

###语言合成
@app.route('/api/custom_languages', methods=['GET'])
def get_custom_languages():
    try:
        inference_files = [f for f in os.listdir(VITS_INFERENCE_DIR) if f.endswith('_inference.py')]
        languages = ['zhongwen', 'sichuan'] + [f.replace('_inference.py', '') for f in inference_files if f not in ['zhongwen_inference.py', 'sichuan_inference.py']]
        return jsonify({"languages": languages})
    except Exception as e:
        return jsonify({"error": f"获取语言列表失败: {str(e)}"}), 500

@app.route('/api/save_language', methods=['POST'])
def save_custom_language():
    data = request.get_json()
    language = data.get('language')
    code = data.get('code')
    delete = data.get('delete', False)

    if not language:
        return jsonify({"error": "缺少语言名称"}), 400

    inference_file = os.path.join(VITS_INFERENCE_DIR, f"{language}_inference.py")

    if delete:
        if language in ['zhongwen', 'sichuan']:
            return jsonify({"error": "不能删除默认语言 zhongwen 或 sichuan"}), 400
        if os.path.exists(inference_file):
            try:
                os.remove(inference_file)
                return jsonify({"message": f"语言 {language} 已删除"})
            except Exception as e:
                return jsonify({"error": f"删除语言失败: {str(e)}"}), 500
        return jsonify({"error": f"语言 {language} 不存在"}), 400

    if not code:
        return jsonify({"error": "缺少推理代码"}), 400

    try:
        with open(inference_file, 'w', encoding='utf-8') as f:
            f.write(code)
        return jsonify({"message": f"语言 {language} 保存成功", "file_path": inference_file})
    except Exception as e:
        return jsonify({"error": f"保存语言失败: {str(e)}"}), 500

@app.route('/api/synthesize_speech', methods=['POST'])
def synthesize_speech():
    if 'model' not in request.files or 'config' not in request.files:
        return jsonify({"error": "缺少模型文件或配置文件"}), 400
    model_file = request.files['model']
    config_file = request.files['config']
    language = request.form.get('language')
    text = request.form.get('text')
    speaker = request.form.get('speaker', '0')
    length_scale = request.form.get('length_scale', '1.0')
    noise_scale = request.form.get('noise_scale', '0.3')
    noise_scale_w = request.form.get('noise_scale_w', '0.5')
    use_pinyin = request.form.get('use_pinyin', 'false').lower() == 'true'
    pitch_factor = float(request.form.get('pitch_factor', '1.0'))

    if not language or not text:
        return jsonify({"error": "缺少语言或合成文本"}), 400
    if not model_file.filename or not config_file.filename:
        return jsonify({"error": "未选择模型文件或配置文件"}), 400
    if not allowed_file(model_file.filename, {'pth'}) or not allowed_file(config_file.filename, {'json'}):
        return jsonify({"error": "文件格式不支持，仅支持 .pth 和 .json"}), 400

    # 保存上传的文件
    model_filename = secure_filename(model_file.filename)
    config_filename = secure_filename(config_file.filename)
    model_path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4().hex}_{model_filename}")
    config_path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4().hex}_{config_filename}")
    model_file.save(model_path)
    config_file.save(config_path)

    inference_script = os.path.join(VITS_INFERENCE_DIR, f"{language}_inference.py")
    if not os.path.exists(inference_script):
        return jsonify({"error": f"推理脚本 {language}_inference.py 不存在"}), 400

    conda_env_path = os.path.join(os.path.expanduser("~"), "anaconda3", "envs", "VITS_train_env")
    python_executable = os.path.join(conda_env_path, "bin", "python")

    if not os.path.exists(python_executable):
        return jsonify({"error": f"Python 可执行文件未找到: {python_executable}"}), 500

    unique_id = str(uuid.uuid4())[:8]
    output_filename = f"{uuid.uuid4().hex}.wav"
    output_path = os.path.join(OUTPUT_DIR, output_filename)

    cmd = [
        python_executable,
        inference_script,
        "-m", model_path,
        "-c", config_path,
        "-t", text,
        "-s", speaker,
        "-l", language,
        "-ls", str(length_scale),
        "--noise_scale", str(float(noise_scale) * pitch_factor),
        "--noise_scale_w", str(float(noise_scale_w) * pitch_factor),
        "--output", output_path
    ]
    if use_pinyin:
        cmd.append("--use_pinyin")

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PATH"] = f"{os.path.join(conda_env_path, 'bin')}:{env['PATH']}"
    env["LD_LIBRARY_PATH"] = f"{os.path.join(conda_env_path, 'lib')}:/usr/lib/x86_64-linux-gnu:{env.get('LD_LIBRARY_PATH', '')}"
    env["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0")

    log_file = os.path.join(BASE_DIR, "synthesis_log.txt")
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Executing synthesis command: {' '.join(cmd)}\n")

    try:
        process = subprocess.run(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=60,
            cwd=VITS_INFERENCE_DIR
        )
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] STDOUT: {process.stdout}\n")
            f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] STDERR: {process.stderr}\n")

        if process.returncode == 0:
            audio_url = f"/outputs/{output_filename}"
            return jsonify({"message": "语音合成成功", "audio_url": audio_url})
        else:
            error_msg = f"语音合成失败，退出码 {process.returncode}: {process.stderr}"
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {error_msg}\n")
            return jsonify({"error": error_msg, "raw_output": process.stderr}), 500
    except subprocess.TimeoutExpired:
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 语音合成超时\n")
        return jsonify({"error": "语音合成超时"}), 500
    except Exception as e:
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 语音合成失败: {str(e)}\n")
        return jsonify({"error": f"语音合成失败: {str(e)}"}), 500

@app.route('/outputs/<filename>', methods=['GET'])
def serve_audio(filename):
    return send_from_directory(OUTPUT_DIR, filename)

###实验项目
# --- 新增实验项目相关路由 ---
@app.route('/api/user', methods=['GET'])
@require_auth()
def get_user():
    return jsonify({
        'username': request.user['username'],
        'role': request.user['role']
    })

# 获取项目列表
@app.route('/api/projects', methods=['GET'])
@require_auth()
def get_projects():
    username = request.user['username']
    user_dir = USER_DATA_ROOT / username

    result = []
    if not user_dir.exists():
        return jsonify(result)

    for pid_dir in user_dir.iterdir():
        if pid_dir.is_dir():
            json_file = pid_dir / "project.json"
            if json_file.is_file():
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    result.append({
                        "e_id": data["e_id"],
                        "e_name": data["e_name"],
                        "e_type": data["e_type"],
                        "e_level": data["e_level"],
                        "e_description": data["e_description"],
                        "e_list": data["e_list"]
                    })
                except Exception as e:
                    logger.warning(f"读取项目失败 {pid_dir}: {e}")
                    continue

    # 按 e_id 时间戳倒序（越大越新）
    result.sort(key=lambda x: x["e_id"], reverse=True)
    return jsonify(result)

@app.route('/api/projects/<project_id>/json', methods=['GET'])
@require_auth()
def get_project_json(project_id):
    username = request.user['username']
    user_dir = USER_DATA_ROOT / username
    project_dir = user_dir / project_id
    json_file = project_dir / "project.json"

    if not json_file.exists():
        return jsonify({'error': '项目文件不存在'}), 404

    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return Response(
            json.dumps(data, ensure_ascii=False, indent=2),
            mimetype='application/json',
            headers={
                'Content-Disposition': f'inline; filename="{project_id}.json"'
            }
        )
    except Exception as e:
        logger.error(f"读取项目 JSON 失败 {project_id}: {e}")
        return jsonify({'error': '读取失败'}), 500

@app.route('/api/projects/<project_id>/guide', methods=['PUT'])
@require_auth('admin')
def save_project_guide(project_id):
    data = request.get_json()
    steps = data.get('steps', [])
    
    if not isinstance(steps, list):
        return jsonify({'error': 'steps 必须是数组'}), 400

    username = request.user['username']
    user_dir = USER_DATA_ROOT / username
    project_dir = user_dir / project_id
    json_file = project_dir / "project.json"

    if not json_file.exists():
        return jsonify({'error': '项目不存在'}), 404

    try:
        # 读取原有数据
        with open(json_file, 'r', encoding='utf-8') as f:
            project_data = json.load(f)

        # 覆盖 e_list
        project_data['e_list'] = steps

        # 写回文件
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(project_data, f, ensure_ascii=False, indent=2)

        return jsonify({'message': '指导书保存成功', 'e_list': steps}), 200
    except Exception as e:
        logger.error(f"保存指导书失败 {project_id}: {e}")
        return jsonify({'error': '保存失败'}), 500

# 创建项目
@app.route('/api/projects', methods=['POST'])
@require_auth('admin')
def create_project():
    data = request.get_json()
    name        = data.get('name')
    project_type= data.get('type')      # 英文
    level       = data.get('level')     # 英文
    description = data.get('description', '')
    
    if not all([name, project_type, level]):
        return jsonify({'error': '缺少必要字段'}), 400

    username   = request.user['username']
    user_dir   = USER_DATA_ROOT / username
    user_dir.mkdir(parents=True, exist_ok=True)

    # 生成唯一项目 ID
    project_id = f"{int(time.time()*1000)}_{uuid.uuid4().hex[:8]}"
    project_dir = user_dir / project_id
    project_dir.mkdir(exist_ok=True)

    # 转为中文
    e_type = TYPE_MAP.get(project_type, project_type)
    e_level = LEVEL_MAP.get(level, level)

    # 生成 project.json 的访问链接
    banner_url = f"http://192.168.1.124:5000/api/projects/{project_id}/json"

    # 构造新格式的 JSON 数据
    project_info = {
        "e_id": project_id,
        "e_name": name,
        "e_type": e_type,
        "e_level": e_level,
        "e_description": description,
        "e_banner": banner_url,
        "e_list": []
    }

    json_path = project_dir / "project.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(project_info, f, ensure_ascii=False, indent=2)

    return jsonify({"e_id": project_id}), 201

# 更新项目
@app.route('/api/projects/<project_id>', methods=['PUT'])
@require_auth('admin')
def update_project(project_id):
    data = request.get_json()
    username = request.user['username']
    json_path = USER_ROOT / username / project_id / "project.json"

    if not json_path.is_file():
        return jsonify({'error': '项目不存在'}), 404

    with open(json_path, 'r', encoding='utf-8') as f:
        info = json.load(f)

    # 只允许改这三个字段
    info['name']        = data.get('name', info['name'])
    info['type']        = data.get('type', info['type'])
    info['level']       = data.get('level', info['level'])
    info['description'] = data.get('description', info.get('description', ''))

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(info, f, ensure_ascii=False, indent=2)

    return jsonify({'message': '项目更新成功'})

# 删除项目
@app.route('/api/projects/<project_id>', methods=['DELETE'])
@require_auth('admin')
def delete_project(project_id):
    username = request.user['username']
    user_dir = USER_DATA_ROOT / username
    project_dir = user_dir / project_id

    if not project_dir.exists() or not project_dir.is_dir():
        return jsonify({'error': '项目不存在'}), 404

    try:
        shutil.rmtree(project_dir)  # 递归删除整个文件夹
        return jsonify({'message': '项目删除成功'}), 200
    except Exception as e:
        logger.error(f"删除项目失败 {project_id}: {e}")
        return jsonify({'error': '删除失败，请检查权限或文件占用'}), 500

# 保存指导书
@app.route('/api/guides', methods=['POST'])
@require_auth('admin')
def save_guide():
    data = request.get_json()
    project_id = data.get('project_id')
    steps = data.get('steps')
    if not project_id or not steps:
        return jsonify({'error': '缺少必要字段'}), 400

    try:
        with get_db() as conn:
            c = conn.cursor()
            c.execute('SELECT name, type, level, created_at FROM projects WHERE id = ?', (project_id,))
            row = c.fetchone()
            if not row:
                return jsonify({'error': '项目不存在'}), 404
            proj_name, proj_type, proj_level, created_at = row

        # 组装 JSON 数据（包含 created_at）
        guide_json = {
            "project_name": proj_name,
            "project_type": proj_type,
            "project_level": proj_level,
            "created_at": created_at,   # 新增字段
            "steps": steps
        }

        # 保存到用户目录下的 project_id 文件夹
        username = request.user['username']
        user_dir = Path(__file__).with_name('User') / username
        project_dir = user_dir / str(project_id)
        project_dir.mkdir(parents=True, exist_ok=True)

        json_path = project_dir / "guide.json"

        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(guide_json, f, ensure_ascii=False, indent=2)

        # 更新数据库中的 guide_path
        with get_db() as conn:
            c = conn.cursor()
            c.execute('UPDATE projects SET guide_path = ? WHERE id = ?', (str(json_path), project_id))
            conn.commit()

        return jsonify({
            'message': '指导书保存成功',
            'guide_path': str(json_path)
        })

    except Exception as e:
        logger.exception('save_guide error')
        return jsonify({'error': str(e)}), 500

@app.route('/api/guides/<path:filename>')
def serve_guide(filename):
    return send_from_directory(GUIDE_DIR, filename)

# 下载指导书 JSON（回显用）
@app.route('/api/guides/<int:project_id>/content', methods=['GET'])
@require_auth()
def get_guide_content(project_id):
    try:
        with get_db() as conn:
            c = conn.cursor()
            c.execute('SELECT guide_path FROM projects WHERE id = ?', (project_id,))
            row = c.fetchone()
            if not row or not row['guide_path']:
                return jsonify({'error': '指导书尚未创建'}), 404
            json_path = Path(row['guide_path'])
            if not json_path.exists():
                return jsonify({'error': '指导书文件丢失'}), 404

        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return jsonify(data)          # {project_name, project_type, project_level, steps}
    except Exception as e:
        logger.exception('get_guide_content error')
        return jsonify({'error': str(e)}), 500

# 获取学生IP
@app.route('/api/student_ips', methods=['GET'])
@require_auth('admin')
def get_student_ips():
    try:
        if not STUDENT_IPS_FILE.exists():
            return jsonify({'ips': []})
        with open(STUDENT_IPS_FILE, 'r', encoding='utf-8') as f:
            ips = [line.strip() for line in f if line.strip()]
        return jsonify({'ips': ips})
    except Exception as e:
        logger.error(f"获取学生IP失败: {str(e)}")
        return jsonify({'ips': []})

# 下发项目
@app.route('/api/projects/<int:id>/distribute', methods=['POST'])
@require_auth('admin')
def distribute_project(id):
    data = request.get_json()
    ips = data.get('ips', [])
    if not ips:
        return jsonify({'error': '未选择学生IP'}), 400
    with get_db() as conn:
        c = conn.cursor()
        c.execute('SELECT id, name, type, level, created_at, guide_path FROM projects WHERE id = ?', (id,))
        project = c.fetchone()
        if not project:
            return jsonify({'error': '项目不存在'}), 404
        for ip in ips:
            c.execute('INSERT OR REPLACE INTO distributed_projects (project_id, student_ip) VALUES (?, ?)', (id, ip))
            # ✅ 用“账号名”代替 IP 当目录名
            username = request.user['username']          # ← 新增
            student_dir = Path(__file__).with_name('User') / username   # ← 改这里
            student_dir.mkdir(parents=True, exist_ok=True)
            dst_path = student_dir / f'guide_{id}.json'
            src_path = Path(project['guide_path'])
            if src_path.exists():
                shutil.copyfile(src_path, dst_path)
            else:
                empty = {
                    "project_name": project['name'],
                    "project_type": project['type'],
                    "project_level": project['level'],
                    "steps": [{"title": "暂未设计步骤", "content": ""}]
                }
                with open(dst_path, 'w', encoding='utf-8') as f:
                    json.dump(empty, f, ensure_ascii=False, indent=2)
            socketio.emit('project_distributed', {
                'id': project['id'],
                'name': project['name'],
                'type': project['type'],
                'level': project['level'],
                'created_at': project['created_at'],
                'guide_path': str(dst_path)          # ← 学生端用新路径
            }, to=ip)
        conn.commit()
        return jsonify({'message': '项目下发成功'})

# 学生端读取自己目录下的指导书
@app.route('/api/student/guide/<int:project_id>', methods=['GET'])
@require_auth()
def get_student_guide(project_id):
    try:
        username = request.user['username']          # 学生账号
        student_dir = Path(__file__).with_name('User') / username
        guide_path = student_dir / f'guide_{project_id}.json'

        if not guide_path.exists():
            return jsonify({'error': '指导书未下发或不存在'}), 404

        with open(guide_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return jsonify(data)
    except Exception as e:
        logger.exception('get_student_guide error')
        return jsonify({'error': str(e)}), 500

@app.route('/api/receive-guide', methods=['POST'])
def receive_guide():
    data = request.get_json()
    e_id = data.get('e_id')
    guide_content = data.get('guide')

    if not e_id or not guide_content:
        return jsonify({'error': '缺少 e_id 或 guide'}), 400

    username = "student"  # 模拟学生
    base_dir = Path("/data/huangtianle/Ai-Voice-Platform-student") / "User" / username
    project_dir = base_dir / e_id
    project_dir.mkdir(parents=True, exist_ok=True)

    json_path = project_dir / "project.json"

    try:
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(guide_content, f, ensure_ascii=False, indent=2)
        logging.info(f"实验下发成功: {username}/{e_id}")
        return jsonify({'message': '实验下发成功'}), 200
    except Exception as e:
        logging.error(f"下发失败 {e_id}: {e}")
        return jsonify({'error': '保存失败'}), 500

# === 接收下发项目接口 ===
@app.route('/receive_project', methods=['POST'])
def receive_project():
    global current_username
    data = request.get_json()
    project_id = data.get('project_id')

    if not project_id:
        return jsonify({'error': '缺少 project_id'}), 400

    if not current_username:
        return jsonify({'error': '学生端未登录，无用户名'}), 400

    try:
        # 关键：保存到用户名目录
        base_dir = Path('/data/huangtianle/Ai-Voice-Platform-student/User')
        save_dir = base_dir / current_username / f"{project_id}"
        save_dir.mkdir(parents=True, exist_ok=True)
        
        (save_dir / 'project.json').write_text(
            json.dumps({'project_id': project_id, 'username': current_username}, ensure_ascii=False),
            encoding='utf-8'
        )
        
        print(f"[学生端] 项目 {project_id} 已下发给用户 {current_username}，保存至 {save_dir}")
        return jsonify({'message': '接收成功'}), 200
    except Exception as e:
        print(f"[学生端] 保存失败: {e}")
        return jsonify({'error': str(e)}), 500



# =============  1. 创建客户端实例  =============
sio = Client(reconnection=True, reconnection_attempts=5, reconnection_delay=1)

@sio.on('connect')
def on_connect():
    print(f'[学生端] 已连接主服务器: http://{args.server}')
    sio.emit('register_student', {
        'ip': get_local_ip(),   # 建议改为本机实际 IP
        'port': args.port
    })

@sio.on('disconnect')
def on_disconnect():
    print('[学生端] 与主服务器断开连接')

@sio.on('connect_error')
def on_connect_error(data):
    print(f'[学生端] 连接主服务器失败: {data}')

@sio.on('recycle_project')
def on_recycle_project(data):
    project_id = data.get('project_id')
    if not project_id:
        print("[回收实验] 缺少 project_id")
        return
    try:
        base_dir = Path('/data/huangtianle/Ai-Voice-Platform-student/User')
        project_dir = base_dir / current_username / project_id
        if project_dir.exists() and project_dir.is_dir():
            shutil.rmtree(project_dir)
            print(f"[回收实验] 已删除项目目录: {project_dir}")
        else:
            print(f"[回收实验] 项目目录不存在: {project_dir}")
    except Exception as e:
        print(f"[回收实验] 删除失败: {e}")

@sio.on('*')
def catch_all(event, data):
    print(f'[学生端] 收到事件 {event} 数据 {data}')

current_username = None

# ==================== 主程序入口（关键修复） ====================
if __name__ == '__main__':
    import argparse
    import eventlet
    eventlet.monkey_patch()

    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=5001,
                        help='学生端本地端口（Flask 服务监听端口）')
    parser.add_argument('--server', type=str, default='127.0.0.1:5000',
                        help='主服务器地址，格式 ip:port')
    parser.add_argument('--username', type=str, required=True,
                        help='当前登录的学生用户名')
    args = parser.parse_args()
    current_username = args.username

    # 启动 Flask 服务（子线程）
    def run_flask():
        print(f"[学生端] 正在启动 Flask 服务于 http://0.0.0.0:{args.port}")
        socketio.run(app,
                     host='0.0.0.0',
                     port=args.port,
                     debug=False,
                     use_reloader=False)

    flask_thread = threading.Thread(target=run_flask, daemon=False)
    flask_thread.start()

    # 等待 Flask 启动
    time.sleep(1.5)

    # 连接主服务器 WebSocket
    try:
        connect_url = f'http://{args.server}'
        print(f"[学生端] 正在连接主服务器: {connect_url}")
        sio.connect(connect_url)
        print(f'[学生端] 已连接主服务器: {connect_url}')
    except Exception as e:
        print(f'[学生端] 首次连接失败: {e}，稍后自动重试...')

    # 主线程阻塞，保持连接
    sio.wait()