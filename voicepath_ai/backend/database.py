"""
SQLite database models for VoicePath AI.
Users (login credentials) + Patients (health profile) + VoiceResults (history).
"""
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime

db = SQLAlchemy()


class User(db.Model):
    __tablename__ = 'users'
    id            = db.Column(db.Integer, primary_key=True)
    email         = db.Column(db.String(120), unique=True, nullable=False, index=True)
    username      = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    role          = db.Column(db.String(20), default='patient')   # 'patient' | 'doctor' | 'admin'
    created_at    = db.Column(db.DateTime, default=datetime.utcnow)
    is_active     = db.Column(db.Boolean, default=True)

    patient       = db.relationship('Patient', backref='user', uselist=False, cascade='all,delete-orphan')
    voice_results = db.relationship('VoiceResult', backref='user', cascade='all,delete-orphan')

    def set_password(self, pw):
        self.password_hash = generate_password_hash(pw)

    def check_password(self, pw):
        return check_password_hash(self.password_hash, pw)

    def to_dict(self):
        return {'id': self.id, 'email': self.email, 'username': self.username,
                'role': self.role, 'created_at': self.created_at.isoformat()}


class Patient(db.Model):
    __tablename__ = 'patients'
    id              = db.Column(db.Integer, primary_key=True)
    user_id         = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False, unique=True)

    # Personal info
    full_name       = db.Column(db.String(120), nullable=False)
    age             = db.Column(db.Integer)
    gender          = db.Column(db.String(20))         # Male / Female / Other
    date_of_birth   = db.Column(db.String(20))
    phone           = db.Column(db.String(30))
    address         = db.Column(db.String(300))

    # Medical history
    has_diabetes    = db.Column(db.Boolean, default=False)
    diabetes_type   = db.Column(db.String(30))         # Type 1 / Type 2 / Pre-diabetic
    has_hypertension= db.Column(db.Boolean, default=False)
    has_asthma      = db.Column(db.Boolean, default=False)
    has_heart_disease=db.Column(db.Boolean, default=False)
    smoker          = db.Column(db.Boolean, default=False)
    alcohol_use     = db.Column(db.Boolean, default=False)
    family_history  = db.Column(db.Text)               # free text
    current_medications = db.Column(db.Text)
    allergies       = db.Column(db.Text)
    notes           = db.Column(db.Text)

    updated_at      = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def to_dict(self):
        return {
            'id': self.id, 'user_id': self.user_id,
            'full_name': self.full_name, 'age': self.age,
            'gender': self.gender, 'date_of_birth': self.date_of_birth,
            'phone': self.phone, 'address': self.address,
            'has_diabetes': self.has_diabetes, 'diabetes_type': self.diabetes_type,
            'has_hypertension': self.has_hypertension,
            'has_asthma': self.has_asthma,
            'has_heart_disease': self.has_heart_disease,
            'smoker': self.smoker, 'alcohol_use': self.alcohol_use,
            'family_history': self.family_history,
            'current_medications': self.current_medications,
            'allergies': self.allergies, 'notes': self.notes,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
        }


class VoiceResult(db.Model):
    __tablename__ = 'voice_results'
    id          = db.Column(db.Integer, primary_key=True)
    user_id     = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    disease     = db.Column(db.String(50), nullable=False)
    label       = db.Column(db.Integer)                # 0=HC, 1=disease
    label_name  = db.Column(db.String(80))
    confidence  = db.Column(db.Float)
    uncertain   = db.Column(db.Boolean, default=False)
    features    = db.Column(db.Text)                   # JSON string
    created_at  = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self):
        import json
        return {
            'id': self.id, 'disease': self.disease,
            'label': self.label, 'label_name': self.label_name,
            'confidence': self.confidence, 'uncertain': self.uncertain,
            'features': json.loads(self.features) if self.features else {},
            'created_at': self.created_at.isoformat(),
        }
