from functools import wraps
from flask import Blueprint, render_template, request, redirect, url_for, flash, session
from werkzeug.security import generate_password_hash
from models import db, User, SearchHistory

admin_bp = Blueprint('admin', __name__, url_prefix='/admin')

# 1. Custom Admin Access Decorator
def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get('admin_authenticated'):
            flash('Please log in with admin credentials to access this page.', 'warning')
            return redirect(url_for('admin.login'))
        return f(*args, **kwargs)
    return decorated_function

# 2. Admin Authentication Routes
@admin_bp.route('/login', methods=['GET', 'POST'])
def login():
    if session.get('admin_authenticated'):
        return redirect(url_for('admin.dashboard'))

    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '').strip()

        # Enforce your required admin credentials
        if username == 'admin123' and password == 'admin123':
            session['admin_authenticated'] = True
            session['admin_user'] = 'admin123'
            flash('Admin authentication successful.', 'success')
            return redirect(url_for('admin.dashboard'))
        else:
            flash('Invalid admin credentials. Please try again.', 'danger')

    return render_template('admin_login.html')

@admin_bp.route('/logout')
def logout():
    session.pop('admin_authenticated', None)
    session.pop('admin_user', None)
    flash('Logged out of admin panel.', 'info')
    return redirect(url_for('admin.login'))

# 3. Read Database (Dashboard)
@admin_bp.route('/')
@admin_required
def dashboard():
    users = User.query.order_by(User.id.desc()).all()
    search_histories = SearchHistory.query.order_by(SearchHistory.timestamp.desc()).all()
    total_users = len(users)
    total_searches = len(search_histories)

    return render_template(
        'admin_dashboard.html',
        users=users,
        search_histories=search_histories,
        total_users=total_users,
        total_searches=total_searches
    )

# 4. User CRUD Operations
@admin_bp.route('/user/create', methods=['POST'])
@admin_required
def create_user():
    username = request.form.get('username', '').strip()
    email = request.form.get('email', '').strip()
    password = request.form.get('password', '').strip()
    location = request.form.get('location', 'India').strip()

    if not username or not email or not password:
        flash('Username, email, and password are required.', 'danger')
        return redirect(url_for('admin.dashboard'))

    if User.query.filter((User.username == username) | (User.email == email)).first():
        flash('User with that username or email already exists.', 'warning')
        return redirect(url_for('admin.dashboard'))

    new_user = User(
        username=username,
        email=email,
        password_hash=generate_password_hash(password),
        location=location
    )
    db.session.add(new_user)
    db.session.commit()
    flash(f'User "{username}" created successfully.', 'success')
    return redirect(url_for('admin.dashboard'))

@admin_bp.route('/user/<int:user_id>/update', methods=['POST'])
@admin_required
def update_user(user_id):
    user = User.query.get_or_404(user_id)
    username = request.form.get('username', '').strip()
    email = request.form.get('email', '').strip()
    location = request.form.get('location', '').strip()
    new_password = request.form.get('password', '').strip()

    if username:
        user.username = username
    if email:
        user.email = email
    if location:
        user.location = location
    if new_password:
        user.password_hash = generate_password_hash(new_password)

    db.session.commit()
    flash(f'User ID {user_id} updated successfully.', 'success')
    return redirect(url_for('admin.dashboard'))

@admin_bp.route('/user/<int:user_id>/delete', methods=['POST'])
@admin_required
def delete_user(user_id):
    user = User.query.get_or_404(user_id)
    # Delete associated search histories
    SearchHistory.query.filter_by(user_id=user.id).delete()
    db.session.delete(user)
    db.session.commit()
    flash(f'User ID {user_id} and associated search logs deleted.', 'success')
    return redirect(url_for('admin.dashboard'))

# 5. Search History CRUD Operations
@admin_bp.route('/history/<int:history_id>/update', methods=['POST'])
@admin_required
def update_history(history_id):
    history_item = SearchHistory.query.get_or_404(history_id)
    updated_query = request.form.get('query_text', '').strip()

    if updated_query:
        history_item.query_text = updated_query
        db.session.commit()
        flash(f'Search entry #{history_id} modified.', 'success')
    else:
        flash('Query text cannot be blank.', 'danger')

    return redirect(url_for('admin.dashboard'))

@admin_bp.route('/history/<int:history_id>/delete', methods=['POST'])
@admin_required
def delete_history(history_id):
    history_item = SearchHistory.query.get_or_404(history_id)
    db.session.delete(history_item)
    db.session.commit()
    flash(f'Search entry #{history_id} deleted successfully.', 'success')
    return redirect(url_for('admin.dashboard'))