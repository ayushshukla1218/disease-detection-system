"""Auth routes: register, login, logout, get current user."""
from flask import Blueprint, request, jsonify, session
from database import db, User, Patient
import re

auth_bp = Blueprint('auth', __name__)


def _validate_email(email):
    return re.match(r'^[^@\s]+@[^@\s]+\.[^@\s]+$', email) is not None


@auth_bp.route('/api/auth/register', methods=['POST'])
def register():
    data = request.get_json()
    if not data:
        return jsonify({'error': 'No data provided'}), 400

    email    = (data.get('email') or '').strip().lower()
    username = (data.get('username') or '').strip()
    password = data.get('password') or ''
    full_name= (data.get('full_name') or '').strip()
    age      = data.get('age')
    gender   = data.get('gender', '')

    if not email or not _validate_email(email):
        return jsonify({'error': 'Valid email required'}), 400
    if not username or len(username) < 3:
        return jsonify({'error': 'Username must be at least 3 characters'}), 400
    if not password or len(password) < 6:
        return jsonify({'error': 'Password must be at least 6 characters'}), 400
    if not full_name:
        return jsonify({'error': 'Full name required'}), 400

    if User.query.filter_by(email=email).first():
        return jsonify({'error': 'Email already registered'}), 409
    if User.query.filter_by(username=username).first():
        return jsonify({'error': 'Username already taken'}), 409

    user = User(email=email, username=username)
    user.set_password(password)
    db.session.add(user)
    db.session.flush()   # get user.id

    patient = Patient(
        user_id=user.id,
        full_name=full_name,
        age=int(age) if age else None,
        gender=gender,
        has_diabetes=data.get('has_diabetes', False),
        diabetes_type=data.get('diabetes_type', ''),
        has_hypertension=data.get('has_hypertension', False),
        has_asthma=data.get('has_asthma', False),
    )
    db.session.add(patient)
    db.session.commit()

    session['user_id'] = user.id
    return jsonify({'message': 'Registration successful', 'user': user.to_dict(),
                    'patient': patient.to_dict()}), 201


@auth_bp.route('/api/auth/login', methods=['POST'])
def login():
    data = request.get_json()
    if not data:
        return jsonify({'error': 'No data provided'}), 400

    identifier = (data.get('email') or data.get('username') or '').strip().lower()
    password   = data.get('password') or ''

    if not identifier or not password:
        return jsonify({'error': 'Email/username and password required'}), 400

    user = (User.query.filter_by(email=identifier).first() or
            User.query.filter_by(username=identifier.lower()).first())

    if not user or not user.check_password(password):
        return jsonify({'error': 'Invalid credentials'}), 401
    if not user.is_active:
        return jsonify({'error': 'Account disabled'}), 403

    session['user_id'] = user.id
    session.permanent = True

    patient = user.patient
    return jsonify({
        'message': 'Login successful',
        'user': user.to_dict(),
        'patient': patient.to_dict() if patient else None,
    })


@auth_bp.route('/api/auth/logout', methods=['POST'])
def logout():
    session.pop('user_id', None)
    return jsonify({'message': 'Logged out'})


@auth_bp.route('/api/auth/me', methods=['GET'])
def me():
    uid = session.get('user_id')
    if not uid:
        return jsonify({'error': 'Not authenticated'}), 401
    user = db.session.get(User, uid)
    if not user:
        return jsonify({'error': 'User not found'}), 404
    return jsonify({'user': user.to_dict(),
                    'patient': user.patient.to_dict() if user.patient else None})


@auth_bp.route('/api/auth/update-profile', methods=['PUT'])
def update_profile():
    uid = session.get('user_id')
    if not uid:
        return jsonify({'error': 'Not authenticated'}), 401

    user = db.session.get(User, uid)
    if not user:
        return jsonify({'error': 'User not found'}), 404

    data = request.get_json() or {}
    patient = user.patient
    if not patient:
        patient = Patient(user_id=uid, full_name=user.username)
        db.session.add(patient)

    fields = ['full_name','age','gender','date_of_birth','phone','address',
              'has_diabetes','diabetes_type','has_hypertension','has_asthma',
              'has_heart_disease','smoker','alcohol_use','family_history',
              'current_medications','allergies','notes']
    for f in fields:
        if f in data:
            setattr(patient, f, data[f])

    db.session.commit()
    return jsonify({'message': 'Profile updated', 'patient': patient.to_dict()})
