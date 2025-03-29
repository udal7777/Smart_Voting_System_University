import datetime
from app import db
from werkzeug.security import generate_password_hash, check_password_hash

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    enrollment_id = db.Column(db.String(50), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    role = db.Column(db.String(20), nullable=False, default='voter')
    is_verified = db.Column(db.Boolean, default=False)
    face_encoding = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.datetime.utcnow)
    
    # Define relationship with Vote model
    votes = db.relationship('Vote', backref='voter', lazy=True)
    
    def set_password(self, password):
        self.password_hash = generate_password_hash(password)
    
    def check_password(self, password):
        return check_password_hash(self.password_hash, password)
    
    def is_admin(self):
        return self.role == 'admin'
    
    def __repr__(self):
        return f'<User {self.enrollment_id}>'

class Election(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(100), nullable=False)
    description = db.Column(db.Text, nullable=True)
    start_date = db.Column(db.DateTime, nullable=False)
    end_date = db.Column(db.DateTime, nullable=False)
    is_active = db.Column(db.Boolean, default=False)
    created_by = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.datetime.utcnow)
    
    # Define relationship with Candidate model
    candidates = db.relationship('Candidate', backref='election', lazy=True, cascade="all, delete-orphan")
    # Define relationship with Vote model
    votes = db.relationship('Vote', backref='election', lazy=True, cascade="all, delete-orphan")
    
    # Define relationship with User model for the admin who created it
    admin = db.relationship('User', backref='created_elections')
    
    def __repr__(self):
        return f'<Election {self.title}>'

class Candidate(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    enrollment_id = db.Column(db.String(50), nullable=False)
    position = db.Column(db.String(100), nullable=False)
    bio = db.Column(db.Text, nullable=True)
    election_id = db.Column(db.Integer, db.ForeignKey('election.id'), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.datetime.utcnow)
    
    # Define relationship with Vote model
    votes = db.relationship('Vote', backref='candidate', lazy=True)
    
    def __repr__(self):
        return f'<Candidate {self.name} for {self.position}>'

class Vote(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    election_id = db.Column(db.Integer, db.ForeignKey('election.id'), nullable=False)
    candidate_id = db.Column(db.Integer, db.ForeignKey('candidate.id'), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.datetime.utcnow)
    
    # Add a unique constraint to prevent duplicate votes
    __table_args__ = (db.UniqueConstraint('user_id', 'election_id'),)
    
    def __repr__(self):
        return f'<Vote by user {self.user_id} for candidate {self.candidate_id}>'

class VerificationToken(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    token = db.Column(db.String(100), nullable=False, unique=True)
    expires_at = db.Column(db.DateTime, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.datetime.utcnow)
    
    # Define relationship with User model
    user = db.relationship('User', backref='verification_tokens')
    
    def is_expired(self):
        return datetime.datetime.utcnow() > self.expires_at
    
    def __repr__(self):
        return f'<VerificationToken for user {self.user_id}>'
