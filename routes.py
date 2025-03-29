import os
import logging
import uuid
import secrets
import datetime
from flask import render_template, request, redirect, url_for, flash, session, jsonify
from werkzeug.security import generate_password_hash
from app import app, db
from models import User, Election, Candidate, Vote, VerificationToken
import re

# Helper function to check if user is logged in
def is_logged_in():
    return 'user_id' in session

# Helper function to check if logged in user is admin
def is_admin():
    if not is_logged_in():
        return False
    
    user_id = session.get('user_id')
    user = User.query.get(user_id)
    return user and user.is_admin()

# Helper function to validate university email
def is_valid_university_email(email):
    # Specific pattern to match only indoreinstitute.com domain
    indore_institute_pattern = r'^[a-zA-Z0-9._%+-]+@indoreinstitute\.com$'
    return re.match(indore_institute_pattern, email) is not None

# Helper function for verification (placeholder for now)
def generate_verification_url(token):
    base_url = request.host_url.rstrip('/')
    return f"{base_url}/verify-email/{token}"

# Home route
@app.route('/')
def index():
    if is_logged_in():
        user_id = session.get('user_id')
        user = User.query.get(user_id)
        
        if user.is_admin():
            return redirect(url_for('admin_dashboard'))
        else:
            return redirect(url_for('voter_dashboard'))
    
    return render_template('index.html')

# Voter Registration
@app.route('/voter/register', methods=['GET', 'POST'])
def voter_register():
    # Get current time for template
    now = datetime.datetime.utcnow()
    
    if request.method == 'POST':
        enrollment_id = request.form.get('enrollment_id')
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        face_data = request.form.get('face_data')
        
        # Validate input
        if not all([enrollment_id, email, password, confirm_password, face_data]):
            flash('All fields including face capture are required', 'danger')
            return render_template('voter_register.html', now=now)
        
        if password != confirm_password:
            flash('Passwords do not match', 'danger')
            return render_template('voter_register.html', now=now)
        
        # Validate email format
        if not is_valid_university_email(email):
            flash('Please use a valid university email address (@indoreinstitute.com)', 'danger')
            return render_template('voter_register.html', now=now)
        
        # Check if user already exists
        existing_user = User.query.filter((User.enrollment_id == enrollment_id) | (User.email == email)).first()
        if existing_user:
            flash('User with this enrollment ID or email already exists', 'danger')
            return render_template('voter_register.html', now=now)
        
        # Process face data
        from face_utils import get_face_encoding
        face_encoding, error = get_face_encoding(face_data)
        
        if error:
            flash(f'Face registration failed: {error}', 'danger')
            return render_template('voter_register.html', now=now)
        
        # Create new user
        new_user = User(
            enrollment_id=enrollment_id,
            email=email,
            role='voter',
            is_verified=True,  # Auto-verify for now (since we don't have email functionality)
            face_encoding=face_encoding
        )
        new_user.set_password(password)
        
        # Save to database
        try:
            db.session.add(new_user)
            db.session.commit()
            flash('Registration successful! You can now log in.', 'success')
            return redirect(url_for('voter_login'))
        except Exception as e:
            db.session.rollback()
            logging.error(f"Database error: {str(e)}")
            flash('An error occurred during registration. Please try again.', 'danger')
            return render_template('voter_register.html', now=now)
    
    return render_template('voter_register.html', now=now)

# Email verification
@app.route('/verify-email/<token>')
def verify_email(token):
    verification = VerificationToken.query.filter_by(token=token).first()
    
    if not verification:
        flash('Invalid verification link', 'danger')
        return redirect(url_for('index'))
    
    if verification.is_expired():
        flash('Verification link has expired. Please register again.', 'danger')
        return redirect(url_for('voter_register'))
    
    # Verify user
    user = verification.user
    user.is_verified = True
    
    # Remove verification token
    db.session.delete(verification)
    db.session.commit()
    
    flash('Email verified successfully! You can now log in.', 'success')
    return redirect(url_for('voter_login'))

# Voter Login
@app.route('/voter/login', methods=['GET', 'POST'])
def voter_login():
    # Get current time for template
    now = datetime.datetime.utcnow()
    
    if request.method == 'POST':
        enrollment_id = request.form.get('enrollment_id')
        password = request.form.get('password')
        face_data = request.form.get('face_data')
        
        # Validate input
        if not all([enrollment_id, password, face_data]):
            flash('All fields including face capture are required', 'danger')
            return render_template('voter_login.html', now=now)
        
        # Get user
        user = User.query.filter_by(enrollment_id=enrollment_id, role='voter').first()
        if not user:
            flash('Invalid enrollment ID or password', 'danger')
            return render_template('voter_login.html', now=now)
        
        # Check if user is verified
        if not user.is_verified:
            flash('Please verify your email before logging in', 'warning')
            return render_template('voter_login.html', now=now)
        
        # Check password
        if not user.check_password(password):
            flash('Invalid enrollment ID or password', 'danger')
            return render_template('voter_login.html', now=now)
        
        # Check face authentication
        if not user.face_encoding:
            flash('Face authentication data is missing. Please contact support.', 'danger')
            return render_template('voter_login.html', now=now)
        
        # Compare face with stored encoding
        from face_utils import compare_faces
        match, error = compare_faces(user.face_encoding, face_data)
        
        if error:
            flash(f'Face authentication error: {error}', 'danger')
            return render_template('voter_login.html', now=now)
        
        if not match:
            flash('Face authentication failed. Please try again.', 'danger')
            return render_template('voter_login.html', now=now)
        
        # Login successful
        session['user_id'] = user.id
        flash('Login successful!', 'success')
        return redirect(url_for('voter_dashboard'))
    
    return render_template('voter_login.html', now=now)

# Admin Registration
@app.route('/admin/register', methods=['GET', 'POST'])
def admin_register():
    # Get current time for template
    now = datetime.datetime.utcnow()
    
    if request.method == 'POST':
        enrollment_id = request.form.get('enrollment_id')
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        face_data = request.form.get('face_data')
        
        # Validate input
        if not all([enrollment_id, email, password, confirm_password, face_data]):
            flash('All fields including face capture are required', 'danger')
            return render_template('admin_register.html', now=now)
        
        if password != confirm_password:
            flash('Passwords do not match', 'danger')
            return render_template('admin_register.html', now=now)
        
        # Validate email format
        if not is_valid_university_email(email):
            flash('Please use a valid university email address', 'danger')
            return render_template('admin_register.html', now=now)
        
        # Check if user already exists
        existing_user = User.query.filter((User.enrollment_id == enrollment_id) | (User.email == email)).first()
        if existing_user:
            flash('User with this enrollment ID or email already exists', 'danger')
            return render_template('admin_register.html', now=now)
        
        # Process face data
        from face_utils import get_face_encoding
        face_encoding, error = get_face_encoding(face_data)
        
        if error:
            flash(f'Face registration failed: {error}', 'danger')
            return render_template('admin_register.html', now=now)
        
        # Create new admin user
        new_admin = User(
            enrollment_id=enrollment_id,
            email=email,
            role='admin',
            is_verified=True,  # Auto-verify for now
            face_encoding=face_encoding
        )
        new_admin.set_password(password)
        
        # Save to database
        try:
            db.session.add(new_admin)
            db.session.commit()
            flash('Admin registration successful! You can now log in.', 'success')
            return redirect(url_for('admin_login'))
        except Exception as e:
            db.session.rollback()
            logging.error(f"Database error: {str(e)}")
            flash('An error occurred during registration. Please try again.', 'danger')
            return render_template('admin_register.html', now=now)
    
    return render_template('admin_register.html', now=now)

# Admin Login
@app.route('/admin/login', methods=['GET', 'POST'])
def admin_login():
    # Get current time for template
    now = datetime.datetime.utcnow()
    
    if request.method == 'POST':
        enrollment_id = request.form.get('enrollment_id')
        password = request.form.get('password')
        face_data = request.form.get('face_data')
        
        # Validate input
        if not all([enrollment_id, password, face_data]):
            flash('All fields including face capture are required', 'danger')
            return render_template('admin_login.html', now=now)
        
        # Get admin user
        admin = User.query.filter_by(enrollment_id=enrollment_id, role='admin').first()
        if not admin:
            flash('Invalid enrollment ID or password', 'danger')
            return render_template('admin_login.html', now=now)
        
        # Check if admin is verified
        if not admin.is_verified:
            flash('Please verify your email before logging in', 'warning')
            return render_template('admin_login.html', now=now)
        
        # Check password
        if not admin.check_password(password):
            flash('Invalid enrollment ID or password', 'danger')
            return render_template('admin_login.html', now=now)
        
        # Check face authentication
        if not admin.face_encoding:
            flash('Face authentication data is missing. Please contact support.', 'danger')
            return render_template('admin_login.html', now=now)
        
        # Compare face with stored encoding
        from face_utils import compare_faces
        match, error = compare_faces(admin.face_encoding, face_data)
        
        if error:
            flash(f'Face authentication error: {error}', 'danger')
            return render_template('admin_login.html', now=now)
        
        if not match:
            flash('Face authentication failed. Please try again.', 'danger')
            return render_template('admin_login.html', now=now)
        
        # Login successful
        session['user_id'] = admin.id
        flash('Admin login successful!', 'success')
        return redirect(url_for('admin_dashboard'))
    
    return render_template('admin_login.html', now=now)

# Voter Dashboard
@app.route('/voter/dashboard')
def voter_dashboard():
    if not is_logged_in():
        flash('Please login to access this page', 'warning')
        return redirect(url_for('voter_login'))
    
    user_id = session.get('user_id')
    user = User.query.get(user_id)
    
    if user.is_admin():
        return redirect(url_for('admin_dashboard'))
    
    # Get current time for template and queries
    now = datetime.datetime.utcnow()
    
    # Get active elections
    active_elections = Election.query.filter(
        Election.is_active == True,
        Election.start_date <= now,
        Election.end_date >= now
    ).all()
    
    # Get user's votes
    user_votes = Vote.query.filter_by(user_id=user_id).all()
    voted_election_ids = [vote.election_id for vote in user_votes]
    
    # Filter out elections the user has already voted in
    available_elections = [election for election in active_elections if election.id not in voted_election_ids]
    
    return render_template('voter_dashboard.html', 
                          user=user, 
                          available_elections=available_elections,
                          voted_election_ids=voted_election_ids,
                          now=now)

# Vote in an election
@app.route('/voter/vote/<int:election_id>', methods=['GET', 'POST'])
def vote_in_election(election_id):
    if not is_logged_in():
        flash('Please login to access this page', 'warning')
        return redirect(url_for('voter_login'))
    
    user_id = session.get('user_id')
    user = User.query.get(user_id)
    
    if user.is_admin():
        return redirect(url_for('admin_dashboard'))
    
    # Check if election exists and is active
    election = Election.query.get_or_404(election_id)
    
    if not election.is_active:
        flash('This election is not active', 'warning')
        return redirect(url_for('voter_dashboard'))
    
    # Get current time for date checks and template
    now = datetime.datetime.utcnow()
    
    if now < election.start_date or now > election.end_date:
        flash('This election is not currently open for voting', 'warning')
        return redirect(url_for('voter_dashboard'))
    
    # Check if user has already voted in this election
    existing_vote = Vote.query.filter_by(user_id=user_id, election_id=election_id).first()
    if existing_vote:
        flash('You have already voted in this election', 'warning')
        return redirect(url_for('voter_dashboard'))
    
    # Get candidates for this election
    candidates = Candidate.query.filter_by(election_id=election_id).all()
    
    if request.method == 'POST':
        candidate_id = request.form.get('candidate_id')
        
        if not candidate_id:
            flash('Please select a candidate', 'danger')
            return render_template('vote.html', election=election, candidates=candidates, now=now)
        
        # Verify candidate is valid for this election
        candidate = Candidate.query.get(candidate_id)
        if not candidate or candidate.election_id != election_id:
            flash('Invalid candidate selection', 'danger')
            return render_template('vote.html', election=election, candidates=candidates, now=now)
        
        # Record the vote
        new_vote = Vote(
            user_id=user_id,
            election_id=election_id,
            candidate_id=candidate_id
        )
        
        try:
            db.session.add(new_vote)
            db.session.commit()
            flash('Your vote has been recorded successfully!', 'success')
            return redirect(url_for('voter_dashboard'))
        except Exception as e:
            db.session.rollback()
            logging.error(f"Error recording vote: {str(e)}")
            flash('An error occurred while recording your vote. Please try again.', 'danger')
    
    return render_template('vote.html', election=election, candidates=candidates, now=now)

# Admin Dashboard
@app.route('/admin/dashboard')
def admin_dashboard():
    if not is_logged_in():
        flash('Please login to access this page', 'warning')
        return redirect(url_for('admin_login'))
    
    user_id = session.get('user_id')
    user = User.query.get(user_id)
    
    if not user.is_admin():
        flash('You do not have permission to access this page', 'danger')
        return redirect(url_for('voter_dashboard'))
    
    # Get current time for template
    now = datetime.datetime.utcnow()
    
    # Get all elections created by this admin
    elections = Election.query.filter_by(created_by=user_id).order_by(Election.created_at.desc()).all()
    
    # Get active elections count
    active_count = sum(1 for e in elections if e.is_active and e.end_date >= now)
    
    # Get total candidates count
    total_candidates = Candidate.query.join(Election).filter(Election.created_by == user_id).count()
    
    # Get total votes cast in admin's elections
    total_votes = Vote.query.join(Election).filter(Election.created_by == user_id).count()
    
    return render_template('admin_dashboard.html', 
                          user=user, 
                          elections=elections,
                          active_count=active_count,
                          total_candidates=total_candidates,
                          total_votes=total_votes,
                          now=now)

# Create Election
@app.route('/admin/create-election', methods=['GET', 'POST'])
def create_election():
    if not is_logged_in() or not is_admin():
        flash('You do not have permission to access this page', 'danger')
        return redirect(url_for('admin_login'))
    
    user_id = session.get('user_id')
    
    # Get current time for template
    now = datetime.datetime.utcnow()
    
    if request.method == 'POST':
        title = request.form.get('title')
        description = request.form.get('description')
        start_date_str = request.form.get('start_date')
        end_date_str = request.form.get('end_date')
        
        # Validate input
        if not all([title, start_date_str, end_date_str]):
            flash('Title, start date, and end date are required', 'danger')
            return render_template('create_election.html', now=now)
        
        try:
            # Parse dates
            start_date = datetime.datetime.fromisoformat(start_date_str.replace('T', ' '))
            end_date = datetime.datetime.fromisoformat(end_date_str.replace('T', ' '))
            
            # Validate dates
            if start_date >= end_date:
                flash('End date must be after start date', 'danger')
                return render_template('create_election.html', now=now)
            
            # If start date is in the future, set it to now to make it immediately available
            if start_date > now:
                print(f"Adjusting start date from {start_date} to current time {now}")
                start_date = now
            
            # Create new election
            new_election = Election(
                title=title,
                description=description,
                start_date=start_date,
                end_date=end_date,
                is_active=True,
                created_by=user_id
            )
            
            db.session.add(new_election)
            db.session.commit()
            
            flash('Election created successfully!', 'success')
            return redirect(url_for('manage_candidates', election_id=new_election.id))
        except ValueError:
            flash('Invalid date format', 'danger')
        except Exception as e:
            db.session.rollback()
            logging.error(f"Error creating election: {str(e)}")
            flash('An error occurred while creating the election', 'danger')
    
    return render_template('create_election.html', now=now)

# Manage Candidates
@app.route('/admin/election/<int:election_id>/candidates', methods=['GET', 'POST'])
def manage_candidates(election_id):
    if not is_logged_in() or not is_admin():
        flash('You do not have permission to access this page', 'danger')
        return redirect(url_for('admin_login'))
    
    user_id = session.get('user_id')
    
    # Get election
    election = Election.query.get_or_404(election_id)
    
    # Check if admin owns this election
    if election.created_by != user_id:
        flash('You do not have permission to manage candidates for this election', 'danger')
        return redirect(url_for('admin_dashboard'))
    
    # Get candidates for this election
    candidates = Candidate.query.filter_by(election_id=election_id).all()
    
    # Get current time for template
    now = datetime.datetime.utcnow()
    
    if request.method == 'POST':
        name = request.form.get('name')
        enrollment_id = request.form.get('enrollment_id')
        position = request.form.get('position')
        bio = request.form.get('bio')
        
        # Validate input
        if not all([name, enrollment_id, position]):
            flash('Name, enrollment ID, and position are required', 'danger')
            return render_template('manage_candidates.html', election=election, candidates=candidates, now=now)
        
        # Check if candidate already exists in this election
        existing_candidate = Candidate.query.filter_by(
            enrollment_id=enrollment_id, 
            election_id=election_id
        ).first()
        
        if existing_candidate:
            flash('A candidate with this enrollment ID already exists in this election', 'danger')
            return render_template('manage_candidates.html', election=election, candidates=candidates, now=now)
        
        # Create new candidate
        new_candidate = Candidate(
            name=name,
            enrollment_id=enrollment_id,
            position=position,
            bio=bio,
            election_id=election_id
        )
        
        try:
            db.session.add(new_candidate)
            db.session.commit()
            flash('Candidate added successfully!', 'success')
            return redirect(url_for('manage_candidates', election_id=election_id))
        except Exception as e:
            db.session.rollback()
            logging.error(f"Error adding candidate: {str(e)}")
            flash('An error occurred while adding the candidate', 'danger')
    
    return render_template('manage_candidates.html', election=election, candidates=candidates, now=now)

# Delete Candidate
@app.route('/admin/delete-candidate/<int:candidate_id>', methods=['POST'])
def delete_candidate(candidate_id):
    if not is_logged_in() or not is_admin():
        flash('You do not have permission to access this page', 'danger')
        return redirect(url_for('admin_login'))
    
    user_id = session.get('user_id')
    
    # Get candidate
    candidate = Candidate.query.get_or_404(candidate_id)
    
    # Get election to check ownership
    election = Election.query.get(candidate.election_id)
    
    # Check if admin owns this election
    if election.created_by != user_id:
        flash('You do not have permission to delete this candidate', 'danger')
        return redirect(url_for('admin_dashboard'))
    
    # Check if votes exist for this candidate
    votes_exist = Vote.query.filter_by(candidate_id=candidate_id).first() is not None
    
    if votes_exist:
        flash('Cannot delete candidate as votes have already been cast', 'danger')
        return redirect(url_for('manage_candidates', election_id=election.id))
    
    try:
        db.session.delete(candidate)
        db.session.commit()
        flash('Candidate deleted successfully!', 'success')
    except Exception as e:
        db.session.rollback()
        logging.error(f"Error deleting candidate: {str(e)}")
        flash('An error occurred while deleting the candidate', 'danger')
    
    return redirect(url_for('manage_candidates', election_id=election.id))

# View Election Results
@app.route('/admin/election/<int:election_id>/results')
def view_results(election_id):
    if not is_logged_in() or not is_admin():
        flash('You do not have permission to access this page', 'danger')
        return redirect(url_for('admin_login'))
    
    user_id = session.get('user_id')
    
    # Get election
    election = Election.query.get_or_404(election_id)
    
    # Check if admin owns this election
    if election.created_by != user_id:
        flash('You do not have permission to view results for this election', 'danger')
        return redirect(url_for('admin_dashboard'))
    
    # Get candidates and their vote counts
    candidates = Candidate.query.filter_by(election_id=election_id).all()
    
    results = []
    for candidate in candidates:
        vote_count = Vote.query.filter_by(candidate_id=candidate.id).count()
        results.append({
            'id': candidate.id,
            'name': candidate.name,
            'position': candidate.position,
            'votes': vote_count
        })
    
    # Get total votes
    total_votes = Vote.query.filter_by(election_id=election_id).count()
    
    # Add current time for template
    now = datetime.datetime.utcnow()
    
    return render_template('vote_results.html', 
                          election=election, 
                          results=results,
                          total_votes=total_votes,
                          now=now)

# API: Real-time Vote Counts
@app.route('/api/election/<int:election_id>/results')
def api_election_results(election_id):
    if not is_logged_in() or not is_admin():
        return jsonify({'error': 'Unauthorized'}), 401
    
    user_id = session.get('user_id')
    
    # Get election
    election = Election.query.get_or_404(election_id)
    
    # Check if admin owns this election
    if election.created_by != user_id:
        return jsonify({'error': 'Unauthorized'}), 401
    
    # Get candidates and their vote counts
    candidates = Candidate.query.filter_by(election_id=election_id).all()
    
    results = []
    labels = []
    data = []
    
    for candidate in candidates:
        vote_count = Vote.query.filter_by(candidate_id=candidate.id).count()
        results.append({
            'id': candidate.id,
            'name': candidate.name,
            'position': candidate.position,
            'votes': vote_count
        })
        labels.append(candidate.name)
        data.append(vote_count)
    
    # Get total votes
    total_votes = Vote.query.filter_by(election_id=election_id).count()
    
    # Add current time for template conditions
    now = datetime.datetime.utcnow()
    
    return jsonify({
        'election': {
            'id': election.id,
            'title': election.title,
            'start_date': election.start_date.isoformat(),
            'end_date': election.end_date.isoformat(),
            'is_active': election.is_active,
            'is_ongoing': election.end_date > now
        },
        'results': results,
        'total_votes': total_votes,
        'chart_data': {
            'labels': labels,
            'data': data
        },
        'now': now.isoformat()
    })

# Toggle Election Status
@app.route('/admin/election/<int:election_id>/toggle', methods=['POST'])
def toggle_election(election_id):
    if not is_logged_in() or not is_admin():
        flash('You do not have permission to access this page', 'danger')
        return redirect(url_for('admin_login'))
    
    user_id = session.get('user_id')
    
    # Get election
    election = Election.query.get_or_404(election_id)
    
    # Check if admin owns this election
    if election.created_by != user_id:
        flash('You do not have permission to modify this election', 'danger')
        return redirect(url_for('admin_dashboard'))
    
    # Toggle status
    election.is_active = not election.is_active
    
    try:
        db.session.commit()
        status = "activated" if election.is_active else "deactivated"
        flash(f'Election {status} successfully!', 'success')
    except Exception as e:
        db.session.rollback()
        logging.error(f"Error toggling election status: {str(e)}")
        flash('An error occurred while updating the election status', 'danger')
    
    return redirect(url_for('admin_dashboard'))

# Logout
@app.route('/logout')
def logout():
    session.pop('user_id', None)
    flash('You have been logged out', 'success')
    return redirect(url_for('index'))
