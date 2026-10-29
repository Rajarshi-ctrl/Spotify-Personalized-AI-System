import os
from dotenv import load_dotenv
from flask import Flask
from models import db
from blueprints.auth import auth_bp
from blueprints.main import main_bp
from flask_login import LoginManager
from models import User
from blueprints.admin import admin_bp

load_dotenv()


def create_app():
    app = Flask(__name__)
    app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", "spotify_default_secret")
    app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///spotify_ai.db"
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

    db.init_app(app)

    login_manager = LoginManager()
    login_manager.login_view = "auth.login"
    login_manager.init_app(app)

    @login_manager.user_loader
    def load_user(user_id):
        return User.query.get(int(user_id))

    # Register Blueprints
    app.register_blueprint(auth_bp)
    app.register_blueprint(main_bp)
    app.register_blueprint(admin_bp)

    with app.app_context():
        db.create_all()

    return app

app = create_app()

if __name__ == "__main__":
    app.run(debug=True, port=5000)