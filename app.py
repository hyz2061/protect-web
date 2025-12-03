import sys
import os
import time
import json
import random
from datetime import datetime, timedelta
from flask import Flask, render_template, request, jsonify, session, send_from_directory
from flask_sqlalchemy import SQLAlchemy
import torch
import cv2
import numpy as np
from torchvision import transforms
from sqlalchemy.orm import Session

# ==========================================
# 1. 智能环境与路径配置
# ==========================================
print("--- 正在启动绿动地球后端 ---")
current_dir = os.path.dirname(os.path.abspath(__file__))

possible_code_paths = [
    os.path.join(current_dir, "Code"),
    os.path.join(current_dir, "Garbage-Classification", "Code"),
    r"E:\github\Garbage-Classification\Code"
]

target_code_root = None
for path in possible_code_paths:
    if os.path.exists(path):
        target_code_root = path
        sys.path.append(path)
        print(f"✅ 成功定位 Code 目录: {path}")
        break

# ==========================================
# 2. 自动寻找模型文件
# ==========================================
DataSetInfo_Path = None
Ckpt_Path = None

if target_code_root:
    checkpoint_root = os.path.join(target_code_root, 'checkpoint')
    if os.path.exists(checkpoint_root):
        for root, dirs, files in os.walk(checkpoint_root):
            if 'DataSetInfo.pth' in files:
                DataSetInfo_Path = os.path.join(root, 'DataSetInfo.pth')
                pths = [f for f in files if f.endswith('.pth') and 'DataSet' not in f]
                if 'best.pth' in pths:
                    Ckpt_Path = os.path.join(root, 'best.pth')
                elif 'ckpt_best.pth' in pths:
                    Ckpt_Path = os.path.join(root, 'ckpt_best.pth')
                elif len(pths) > 0:
                    Ckpt_Path = os.path.join(root, sorted(pths)[-1])
                break

if DataSetInfo_Path and Ckpt_Path:
    print(f"✅ 模型路径锁定完成")
else:
    print("❌ 未找到模型文件，AI 功能将不可用")

# ==========================================
# 3. Flask 初始化
# ==========================================
app = Flask(__name__)
app.secret_key = 'garbage_eco_secret_key_2025'
app.config['UPLOAD_FOLDER'] = os.path.join(current_dir, 'static', 'uploads')
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(current_dir, 'database.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
db = SQLAlchemy(app)


# ==========================================
# 4. 数据库模型 (新增 Activity)
# ==========================================
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(120), nullable=False)
    role = db.Column(db.String(20), default='personal', nullable=False)  # personal / store
    points = db.Column(db.Integer, default=0, nullable=False)
    # 关联
    actions = db.relationship('UserAction', backref='user', lazy=True, cascade="all, delete-orphan")
    carbon_records = db.relationship('CarbonRecord', backref='user', lazy=True, cascade="all, delete-orphan")
    activities_created = db.relationship('Activity', backref='creator', lazy=True)


class UserAction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    action_type = db.Column(db.String(50), nullable=False)
    detail = db.Column(db.String(100))
    points_change = db.Column(db.Integer, nullable=False)
    image_path = db.Column(db.String(200))
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)


class CarbonRecord(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    total_carbon = db.Column(db.Float, default=0.0)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)


# [新增] 活动表
class Activity(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    creator_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    title = db.Column(db.String(100), nullable=False)
    description = db.Column(db.String(500))
    date_str = db.Column(db.String(50))  # 简化的日期存储
    location = db.Column(db.String(100))
    image_url = db.Column(db.String(200))  # 活动封面图
    joined_count = db.Column(db.Integer, default=0)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


# ==========================================
# 5. 加载 AI 模型
# ==========================================
print("正在初始化 AI 引擎...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MyModel = None
model_loaded = False
index_to_class = {}
index_to_group = {}

try:
    if target_code_root and DataSetInfo_Path and Ckpt_Path:
        try:
            from Code.models.mobilenetv3 import MobileNetV3_Small
        except ImportError:
            from models.mobilenetv3 import MobileNetV3_Small

        DataSetInfo = torch.load(DataSetInfo_Path, map_location=torch.device('cpu'))

        raw_classes = DataSetInfo.get('index_to_class', [])
        raw_groups = DataSetInfo.get('index_to_group', [])

        if isinstance(raw_classes, list):
            index_to_class = {i: v for i, v in enumerate(raw_classes)}
        else:
            index_to_class = raw_classes

        if isinstance(raw_groups, list):
            index_to_group = {i: v for i, v in enumerate(raw_groups)}
        else:
            index_to_group = raw_groups

        class_num = DataSetInfo.get("class_num", len(index_to_class))
        MyModel = MobileNetV3_Small(class_num)
        checkpoint = torch.load(Ckpt_Path, map_location=device)

        if 'state_dict' in checkpoint:
            MyModel.load_state_dict(checkpoint['state_dict'])
        else:
            MyModel.load_state_dict(checkpoint)

        MyModel.to(device)
        MyModel.eval()
        model_loaded = True
        print(f"✅ AI 引擎就绪 ({class_num}类)")
except Exception as e:
    print(f"❌ 模型加载异常: {e}")


# ==========================================
# 6. AI 推理逻辑
# ==========================================
def run_inference(image_path):
    if not model_loaded or MyModel is None: return "系统维护", "未知", 0
    try:
        cvImg = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if cvImg is None: return "图片无效", "未知", 0

        image = cvImg.copy()
        h, w = image.shape[:2]
        target_size = 224
        ratio = target_size / min(h, w)
        new_h, new_w = int(h * ratio), int(w * ratio)
        if new_h < target_size: new_h = target_size
        if new_w < target_size: new_w = target_size

        image = cv2.resize(image, (new_w, new_h))
        y1 = max(0, (new_h - target_size) // 2)
        x1 = max(0, (new_w - target_size) // 2)
        image = image[y1:y1 + target_size, x1:x1 + target_size]

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        img_tensor = transform(image[:, :, (2, 1, 0)]).unsqueeze(0).to(device)

        with torch.no_grad():
            output = MyModel(img_tensor)
            _, predicted_idx = torch.max(output, 1)
            idx = predicted_idx.item()

        class_name = index_to_class.get(idx, "未知")
        group_name = index_to_group.get(idx, "其他垃圾")

        points = 5
        if '可回收' in group_name:
            points = 10
        elif '有害' in group_name:
            points = 20
        elif '厨余' in group_name or '湿' in group_name:
            points = 5

        return class_name, group_name, points
    except Exception as e:
        print(f"推理错误: {e}")
        return "识别错误", "未知", 0


# ==========================================
# 7. Web 路由接口
# ==========================================
with app.app_context():
    db.create_all()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


# --- 认证 ---
@app.route('/api/register', methods=['POST'])
def register():
    try:
        data = request.json
        if User.query.filter_by(username=data['username']).first():
            return jsonify({'success': False, 'message': '用户已存在'})

        new_user = User(username=data['username'], password=data['password'], role=data.get('role', 'personal'))
        db.session.add(new_user)
        db.session.commit()
        return jsonify({'success': True, 'message': '注册成功'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})


@app.route('/api/login', methods=['POST'])
def login():
    try:
        data = request.json
        user = User.query.filter_by(username=data['username'], password=data['password']).first()
        if user:
            session['user_id'] = user.id
            return jsonify(
                {'success': True, 'user': {'username': user.username, 'points': user.points, 'role': user.role}})
        return jsonify({'success': False, 'message': '账号密码错误'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})


@app.route('/api/logout', methods=['POST'])
def logout():
    session.pop('user_id', None)
    return jsonify({'success': True})


# --- 业务功能 ---
@app.route('/api/identify', methods=['POST'])
def identify():
    if 'user_id' not in session: return jsonify({'success': False, 'message': '请先登录'})
    file = request.files.get('file')
    if not file: return jsonify({'success': False, 'message': '无文件'})

    try:
        filename = f"{int(time.time())}_{file.filename}"
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(save_path)

        c_name, g_name, pts = run_inference(save_path)

        with Session(db.engine) as session_db:
            user = session_db.get(User, session['user_id'])
            user.points += pts
            act = UserAction(user_id=user.id, action_type='identify', detail=f"{g_name}-{c_name}", points_change=pts,
                             image_path=f"/uploads/{filename}")
            session_db.add(act)
            session_db.commit()
            new_points = user.points

        return jsonify({'success': True,
                        'data': {'class_name': c_name, 'group_name': g_name, 'points': pts, 'user_points': new_points}})
    except Exception as e:
        return jsonify({'success': False, 'message': '识别异常'})


@app.route('/api/carbon', methods=['POST'])
def save_carbon():
    if 'user_id' not in session: return jsonify({'success': False})
    data = request.json
    with Session(db.engine) as s:
        user = s.get(User, session['user_id'])
        rec = CarbonRecord(user_id=user.id, total_carbon=float(data.get('total', 0)))
        user.points += 5
        s.add(rec)
        s.add(UserAction(user_id=user.id, action_type='carbon', detail='碳足迹记录', points_change=5))
        s.commit()
        return jsonify({'success': True, 'points': user.points})


# --- [新增] 活动相关接口 ---
@app.route('/api/activities', methods=['GET'])
def get_activities():
    # 获取所有活动，按时间倒序
    acts = Activity.query.order_by(Activity.created_at.desc()).all()
    act_list = []
    for a in acts:
        act_list.append({
            'id': a.id,
            'title': a.title,
            'description': a.description,
            'date': a.date_str,
            'location': a.location,
            'image': a.image_url or "https://images.unsplash.com/photo-1542601906990-b4d3fb7d5fa5?auto=format&fit=crop&w=400",
            'creator': a.creator.username,
            'joined': a.joined_count
        })
    return jsonify({'success': True, 'data': act_list})


@app.route('/api/activities', methods=['POST'])
def create_activity():
    if 'user_id' not in session: return jsonify({'success': False, 'message': '请先登录'})

    with Session(db.engine) as s:
        user = s.get(User, session['user_id'])
        if user.role != 'store':
            return jsonify({'success': False, 'message': '仅认证商家/机构可发布活动'})

        data = request.json
        new_act = Activity(
            creator_id=user.id,
            title=data.get('title'),
            description=data.get('desc'),
            date_str=data.get('date'),
            location=data.get('location'),
            # 这里简单起见，使用一个随机的环保图片，或者可以做图片上传接口
            image_url="https://images.unsplash.com/photo-1526951521990-620dc14c210b?auto=format&fit=crop&w=400"
        )
        s.add(new_act)
        s.commit()

    return jsonify({'success': True, 'message': '活动发布成功'})


@app.route('/api/activities/join', methods=['POST'])
def join_activity():
    if 'user_id' not in session: return jsonify({'success': False, 'message': '请先登录'})
    data = request.json
    act_id = data.get('id')

    with Session(db.engine) as s:
        act = s.get(Activity, act_id)
        if act:
            act.joined_count += 1
            # 记录用户行为
            s.add(UserAction(user_id=session['user_id'], action_type='activity', detail=f"报名:{act.title}",
                             points_change=20))
            user = s.get(User, session['user_id'])
            user.points += 20  # 报名奖励
            s.commit()
            return jsonify({'success': True, 'points': user.points})
    return jsonify({'success': False})


@app.route('/api/leaderboard', methods=['GET'])
def leaderboard():
    # 获取积分前5名
    users = User.query.order_by(User.points.desc()).limit(5).all()
    data = [{'username': u.username, 'points': u.points, 'role': u.role} for u in users]
    return jsonify({'success': True, 'data': data})


# --- 统计与兑换 ---
@app.route('/api/redeem', methods=['POST'])
def redeem():
    if 'user_id' not in session: return jsonify({'success': False})
    data = request.json
    cost = int(data.get('cost', 0))
    item = data.get('item')
    with Session(db.engine) as s:
        user = s.get(User, session['user_id'])
        if user.points < cost: return jsonify({'success': False, 'message': '积分不足'})
        user.points -= cost
        s.add(UserAction(user_id=user.id, action_type='redeem', detail=f"兑换:{item}", points_change=-cost))
        s.commit()
        return jsonify({'success': True, 'points': user.points})


@app.route('/api/stats', methods=['GET'])
def stats():
    if 'user_id' not in session: return jsonify({'success': False})
    user_id = session['user_id']

    with Session(db.engine) as s:
        # 分类统计
        acts = s.query(UserAction).filter_by(user_id=user_id, action_type='identify').all()
        pie_data = [0, 0, 0, 0]
        for a in acts:
            d = a.detail or ""
            if '可回收' in d:
                pie_data[0] += 1
            elif '有害' in d:
                pie_data[1] += 1
            elif '厨余' in d or '湿' in d:
                pie_data[2] += 1
            else:
                pie_data[3] += 1

        # 积分趋势 (最近7条记录)
        points_history = s.query(UserAction).filter_by(user_id=user_id).order_by(UserAction.timestamp.desc()).limit(
            10).all()
        trend_data = [p.points_change for p in points_history][::-1]  # 简单的变化量展示

    return jsonify({'success': True, 'pie': pie_data, 'trend': trend_data})


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)