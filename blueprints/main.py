from flask import Blueprint, render_template, request, redirect, url_for
from flask_login import login_required, current_user
from models import db, SearchHistory
from services.recommender import (
    get_cold_start_tracks,
    get_warm_start_tracks,
    search_catalog_with_ai
)

main_bp = Blueprint("main", __name__)

@main_bp.route("/")
@login_required
def dashboard():
    search_count = SearchHistory.query.filter_by(user_id=current_user.id).count()

    if search_count == 0:
        time_label, cold_tracks = get_cold_start_tracks(current_user.location)
        return render_template(
            "dashboard.html",
            is_cold_start=True,
            time_label=time_label,
            location=current_user.location,
            songs=cold_tracks
        )
    else:
        recent_records = SearchHistory.query.filter_by(user_id=current_user.id)\
            .order_by(SearchHistory.timestamp.desc()).limit(5).all()
        past_queries = [r.query_text for r in recent_records]

        history_songs, time_label, time_songs = get_warm_start_tracks(
            past_queries, current_user.location
        )
        return render_template(
            "dashboard.html",
            is_cold_start=False,
            history_songs=history_songs,
            time_label=time_label,
            time_songs=time_songs
        )

@main_bp.route("/search", methods=["POST"])
@login_required
def search():
    query = request.form.get("search_query", "").strip()
    if not query:
        return redirect(url_for("main.dashboard"))

    # Log search in history
    db.session.add(SearchHistory(user_id=current_user.id, query_text=query))
    db.session.commit()

    # Fetch past queries for personalization context
    past_records = SearchHistory.query.filter_by(user_id=current_user.id)\
        .order_by(SearchHistory.timestamp.desc()).limit(4).all()
    past_queries = [p.query_text for p in past_records]

    matched_tracks, dj_commentary = search_catalog_with_ai(
        query=query,
        location=current_user.location,
        past_queries=past_queries
    )

    return render_template(
        "dashboard.html",
        is_search_result=True,
        search_query=query,
        dj_commentary=dj_commentary,
        search_results=matched_tracks
    )