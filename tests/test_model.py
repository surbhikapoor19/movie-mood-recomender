import os

## Set our local model to be running...
os.environ["LOCAL_MODEL"] = "true"

from app import prepare_request, build_messages, run_local_model

"""
GENRES = ["Horror", "Action", "Thriller", "Comedy", "Science-Fiction", "Drama", "Documentary", "Romance", "Animation"]
MOODS = ["Dark & Intense", "Light & Fun", "Emotional & Deep", "Suspenseful", "Inspirational"]
ERAS = ["Classic", "90s Classics", "2000s", "2010s", "Recent", "Any Era"]
VIEWING_CONTEXTS = ["Solo", "Family", "Friends", "Any"]
PACE_OPTIONS = ["Fast-paced", "Slow & character-driven", "Balanced"]
"""

## Tests that our local model is being called, and returns a something.
def test_local_model_running():
    ## Build a message for testing
    user_message = "Gimme something that reminds me of Bullet Train!"
    genre = "Horror"
    mood = "Light & Fun"
    era = "Classic"
    viewing_pref = "Solo"
    pace = "Balanced"
    ## Create user answers s
    user_answers, _, _ = prepare_request(
            user_message, genre, mood, era, viewing_pref, pace
        )
    full_message = build_messages(user_answers)

    ## Call for our model with a sample input
    response = run_local_model(full_message,max_tokens=50,temperature=.01)

    ## Check that the model gives us a response
    assert response is not "" or None