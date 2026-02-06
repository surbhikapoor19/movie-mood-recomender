import gradio as gr
import json
import re
import os
import requests

# Load environment variables from .env file for local development
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ============================================================================
# CONFIGURATION
# ============================================================================
# Set to True to use local model (transformers pipeline), False for HF Inference API
LOCAL_MODEL = os.environ.get("LOCAL_MODEL", "false").lower() == "true"

# Local model configuration
LOCAL_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"  # Smaller model for local use (~1GB)
# Alternative larger models:
# LOCAL_MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"  # ~3GB
# LOCAL_MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"  # ~14GB

# API model configuration
API_MODEL_NAME = "openai/gpt-oss-120b"

# ============================================================================
# TMDB API CONFIGURATION
# ============================================================================
TMDB_BASE_URL = "https://api.themoviedb.org/3"
TMDB_IMAGE_BASE_URL = "https://image.tmdb.org/t/p/w300"


def get_tmdb_api_key():
    """Get TMDB API key from environment variable."""
    return os.environ.get("TMDB_API_KEY", "")


def search_movie_tmdb(title: str, year: int = None) -> dict | None:
    """Search for a movie on TMDB by title and optionally year."""
    api_key = get_tmdb_api_key()
    if not api_key:
        print(f"[INFO] TMDB: No API key configured, skipping lookup for '{title}'")
        return None
    
    params = {
        "api_key": api_key,
        "query": title,
        "include_adult": False,
    }
    if year:
        params["year"] = year
    
    try:
        print(f"[INFO] TMDB: Searching for '{title}' ({year or 'any year'})...")
        response = requests.get(f"{TMDB_BASE_URL}/search/movie", params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        results = data.get("results", [])
        print(f"[DEBUG] TMDB: Search results: {results}")
        
        if results:
            # Return the first (most relevant) result
            found = results[0]
            print(f"[INFO] TMDB: Found '{found.get('title')}' (rating: {found.get('vote_average', 'N/A')})")
            return found
        print(f"[INFO] TMDB: No results found for '{title}'")
        return None
    except requests.RequestException as e:
        print(f"[WARN] TMDB: Request failed for '{title}': {e}")
        return None


def get_movie_details_tmdb(movie_id: int) -> dict | None:
    """Get detailed movie information from TMDB."""
    api_key = get_tmdb_api_key()
    if not api_key:
        return None
    
    try:
        response = requests.get(
            f"{TMDB_BASE_URL}/movie/{movie_id}",
            params={"api_key": api_key},
            timeout=10
        )
        response.raise_for_status()
        return response.json()
    except requests.RequestException:
        return None


def format_movie_card_with_tmdb(title: str, year: int, why: str, index: int) -> str:
    """Format a movie card, enriching with TMDB data if available."""
    # Try to get TMDB data
    tmdb_data = search_movie_tmdb(title, year)
    
    if tmdb_data:
        # Extract TMDB info
        tmdb_title = tmdb_data.get("title", title)
        tmdb_year = tmdb_data.get("release_date", "")[:4] if tmdb_data.get("release_date") else str(year)
        rating = tmdb_data.get("vote_average", 0)
        overview = tmdb_data.get("overview", "")
        poster_path = tmdb_data.get("poster_path")
        
        # Truncate overview if too long
        if len(overview) > 200:
            overview = overview[:200] + "..."
        
        # Build poster HTML
        if poster_path:
            poster_html = f'<img src="{TMDB_IMAGE_BASE_URL}{poster_path}" alt="{tmdb_title}" style="max-width: 150px; border-radius: 8px; margin-right: 15px;">'
        else:
            # Placeholder for missing poster
            poster_html = '''<div style="width: 150px; min-width: 150px; height: 225px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 8px; margin-right: 15px; display: flex; align-items: center; justify-content: center; color: white; font-size: 48px;">🎬</div>'''
        
        # Format the card with poster
        card = f"""
<div style="display: flex; margin-bottom: 20px; padding: 15px; background: #f8f9fa; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
{poster_html}
<div style="flex: 1;">
<h3 style="margin: 0 0 5px 0; color: #333;">{index}. {tmdb_title} ({tmdb_year})</h3>
<p style="margin: 5px 0;"><strong>Rating:</strong> {rating:.1f}/10</p>
<p style="margin: 5px 0;"><em>{why}</em></p>
<p style="margin: 5px 0; color: #666; font-size: 0.9em;">{overview}</p>
</div>
</div>
"""
        return card
    else:
        # Fallback without TMDB data - styled similarly to TMDB card
        # Placeholder image area
        placeholder_html = '''<div style="width: 150px; min-width: 150px; height: 225px; background: linear-gradient(135deg, #a8a8a8 0%, #6b6b6b 100%); border-radius: 8px; margin-right: 15px; display: flex; align-items: center; justify-content: center; color: white; font-size: 48px;">🎬</div>'''
        
        # Show year if available
        year_display = f" ({year})" if year else ""
        
        return f"""
<div style="display: flex; margin-bottom: 20px; padding: 15px; background: #f8f9fa; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
{placeholder_html}
<div style="flex: 1;">
<h3 style="margin: 0 0 5px 0; color: #333;">{index}. {title}{year_display}</h3>
<p style="margin: 5px 0; color: #888;"><em>Movie details not available from TMDB</em></p>
<p style="margin: 5px 0;"><em>{why}</em></p>
</div>
</div>
"""

# ============================================================================
# MODEL INITIALIZATION
# ============================================================================
local_pipeline = None

if LOCAL_MODEL:
    print(f"[MODE] Loading local model: {LOCAL_MODEL_NAME}")
    try:
        from transformers import pipeline
        import torch
        
        # Force CPU to avoid Metal/MPS memory issues on Mac
        local_pipeline = pipeline(
            "text-generation",
            model=LOCAL_MODEL_NAME,
            device="cpu",  # Use CPU to avoid GPU memory issues
            torch_dtype=torch.float32,
        )
        print(f"[MODE] Local model loaded successfully on CPU!")
    except Exception as e:
        print(f"[ERROR] Failed to load local model: {e}")
        print("[MODE] Falling back to API mode")
        LOCAL_MODEL = False
        from huggingface_hub import InferenceClient

if not LOCAL_MODEL:
    print("[MODE] Using HuggingFace Inference API")
    from huggingface_hub import InferenceClient
    from huggingface_hub import InferenceClient

# ============================================================================
# CUSTOM CSS
# ============================================================================
custom_css = """
/* Overall background */
body, .gradio-container {
    background-color:#EDF3F5;
    font-family: 'Arial', sans-serif;
}
.gradio-container {
    max-width: 700px;
    margin: 0 auto;
    padding: 20px;
    border-radius: 15px;
}
h1 {
    font-style: italic;
    font-weight: bold;
    font-size: 40px;
    text-align: center;
    color: #663356;
    text-shadow: 2px 2px 4px #693256;
}
"""

# ============================================================================
# UI OPTIONS
# ============================================================================
GENRES = ["Horror", "Action", "Thriller", "Comedy", "Science-Fiction", "Drama", "Documentary", "Romance", "Animation"]
MOODS = ["Dark & Intense", "Light & Fun", "Emotional & Deep", "Suspenseful", "Inspirational"]
ERAS = ["Classic", "90s Classics", "2000s", "2010s", "Recent", "Any Era"]
VIEWING_CONTEXTS = ["Solo", "Family", "Friends", "Any"]
PACE_OPTIONS = ["Fast-paced", "Slow & character-driven", "Balanced"]

# ============================================================================
# PROMPT ENGINEERING (from notebook)
# ============================================================================
SYSTEM_PROMPT = """You are a movie recommender. Recommend 3-5 movies matching the user's preferences.

CRITICAL: Output ONLY valid JSON. No text before or after. Follow this EXACT format:

{"user_mentioned_movies":["Movie1"],"recommendations":[{"title":"Movie A","year":2020,"why":"Reason 1"},{"title":"Movie B","year":2019,"why":"Reason 2"},{"title":"Movie C","year":2018,"why":"Reason 3"}]}

Rules:
- The "recommendations" field MUST be a single array containing movie objects
- Each movie object has: "title" (string), "year" (number), "why" (string)
- Include 3-5 movies in the recommendations array
- Prefer well-known movies
- NEVER recommend any movie the user mentions in their message - recommend SIMILAR movies instead
- First extract movies mentioned by the user into "user_mentioned_movies", then recommend DIFFERENT movies
- Provide variety in your recommendations"""

FEW_SHOT_EXAMPLES = [
    {
        "user": {
            "mood": "Dark & Intense",
            "genres": ["Horror", "Thriller"],
            "pace": "Fast-paced",
            "viewing_context": "Solo",
            "era": "Recent",
            "open_ended": "Recently I watched a cool movie called Skinamarink. I like tense movies with clever twists, not gore."
        },
        "assistant": {
            "user_mentioned_movies": ["Skinamarink"],
            "recommendations": [
                {"title": "A Quiet Place", "year": 2018, "why": "Tense survival horror with clever premise and minimal gore."},
                {"title": "Get Out", "year": 2017, "why": "Psychological thriller with sharp twists and social commentary."},
                {"title": "It Follows", "year": 2014, "why": "Atmospheric horror with unique concept and building dread."},
                {"title": "Don't Breathe", "year": 2016, "why": "Intense home invasion thriller with constant suspense."}
            ]
        }
    },
    {
        "user": {
            "mood": "Emotional & Deep",
            "genres": ["Drama"],
            "pace": "Slow & character-driven",
            "viewing_context": "Solo",
            "era": "Classic",
            "open_ended": "I love character growth and bittersweet endings. Kinda like the Shawshank Redemption."
        },
        "assistant": {
            "user_mentioned_movies": ["Shawshank Redemption"],
            "recommendations": [
                {"title": "Good Will Hunting", "year": 1997, "why": "Character-driven drama with emotional growth and warmth."},
                {"title": "Forrest Gump", "year": 1994, "why": "Heartfelt journey through life with bittersweet moments."},
                {"title": "The Green Mile", "year": 1999, "why": "Emotional prison drama with powerful character arcs."},
                {"title": "Dead Poets Society", "year": 1989, "why": "Inspiring story about finding your voice and passion."}
            ]
        }
    },
    {
        "user": {
            "mood": "Light & Fun",
            "genres": ["Comedy", "Animation"],
            "pace": "Fast-paced",
            "viewing_context": "Family",
            "era": "2010s",
            "open_ended": "I loved Spider-Man: Into the Spider-Verse because it was funny, stylish, and heartfelt."
        },
        "assistant": {
            "user_mentioned_movies": ["Spider-Man: Into the Spider-Verse"],
            "recommendations": [
                {"title": "The Lego Movie", "year": 2014, "why": "Colorful animated adventure with humor and heart."},
                {"title": "Big Hero 6", "year": 2014, "why": "Stylish superhero animation with emotional depth."},
                {"title": "Coco", "year": 2017, "why": "Visually stunning with heartfelt family themes."},
                {"title": "The Mitchells vs. the Machines", "year": 2021, "why": "Fast-paced comedy with unique animation style."}
            ]
        }
    }
]

USER_PROMPT_TEMPLATE = """Given the user's answers below, recommend 3-5 movies.

User answers (JSON):
{answers_json}

If the user mentions a movie, pick similar ones. Follow the rules from the system prompt. Output only the JSON object."""


# ============================================================================
# MODEL FUNCTIONS
# ============================================================================
def build_messages(user_answers: dict) -> list[dict]:
    """Assemble system prompt, few-shot examples, and user query into a
    chat-message list ready for the model."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    for ex in FEW_SHOT_EXAMPLES:
        messages.append({
            "role": "user",
            "content": USER_PROMPT_TEMPLATE.format(
                answers_json=json.dumps(ex["user"])
            )
        })
        messages.append({
            "role": "assistant",
            "content": json.dumps(ex["assistant"])
        })

    messages.append({
        "role": "user",
        "content": USER_PROMPT_TEMPLATE.format(
            answers_json=json.dumps(user_answers)
        )
    })

    return messages


def clean_output(raw_content: str) -> str:
    """Extract the assistant's final response, stripping any <think> blocks."""
    clean_out = re.sub(r"<think>.*?</think>\s*", "", raw_content, flags=re.DOTALL)
    return clean_out.strip()


def parse_recommendation(response_text: str) -> dict:
    """Parse the JSON response from the model, handling common malformed outputs."""
    print(f"[INFO] Attempting to parse JSON from response...")
    print(f"\n{'='*60}")
    print(f"[DEBUG] FULL RAW MODEL OUTPUT:")
    print(f"{'='*60}")
    print(response_text)
    print(f"{'='*60}\n")
    
    # First, try direct JSON parsing
    try:
        start = response_text.find('{')
        if start == -1:
            print(f"[WARN] No JSON object found in response")
            return None
        
        # Count braces to find the matching closing brace
        depth = 0
        for i, char in enumerate(response_text[start:], start):
            if char == '{':
                depth += 1
            elif char == '}':
                depth -= 1
                if depth == 0:
                    json_str = response_text[start:i+1]
                    result = json.loads(json_str)
                    print(f"[INFO] Successfully parsed JSON directly")
                    return result
    except json.JSONDecodeError as e:
        print(f"[INFO] Direct JSON parsing failed: {e}")
    
    # Try to fix common malformed JSON issues
    try:
        print(f"[INFO] Attempting to fix malformed JSON...")
        
        # Extract the JSON-like content
        start = response_text.find('{')
        end = response_text.rfind('}') + 1
        if start == -1 or end == 0:
            return None
        
        json_str = response_text[start:end]
        
        # Fix: Multiple arrays instead of objects in array
        # Pattern: [{"title":...}], [{"title":...}] -> {"title":...}, {"title":...}
        json_str = re.sub(r'\],\s*\[', ', ', json_str)
        
        # Fix: Remove extra brackets around individual objects
        # Pattern: [[{...}]] -> [{...}]
        json_str = re.sub(r'\[\[(\{)', r'[\1', json_str)
        json_str = re.sub(r'(\})\]\]', r'\1]', json_str)
        
        # Fix: Trailing commas before closing brackets
        json_str = re.sub(r',\s*\]', ']', json_str)
        json_str = re.sub(r',\s*\}', '}', json_str)
        
        # Fix: Missing commas between objects
        json_str = re.sub(r'\}\s*\{', '}, {', json_str)
        
        # Fix: Unescaped quotes in strings (common issue)
        # This is tricky - try to fix obvious cases
        json_str = re.sub(r'(?<!\\)"(?=\w)', '\\"', json_str)
        
        # Fix: Single quotes to double quotes
        json_str = json_str.replace("'", '"')
        
        print(f"[DEBUG] Fixed JSON (first 500 chars): {json_str[:500]}...")
        
        # Try parsing the fixed JSON
        result = json.loads(json_str)
        print(f"[INFO] Successfully parsed fixed JSON")
        return result
    except json.JSONDecodeError as e:
        print(f"[WARN] Fixed JSON parsing also failed: {e}")
        print(f"[DEBUG] Error at position {e.pos}: ...{json_str[max(0,e.pos-20):e.pos+20]}...")
    
    # Last resort: try to extract movie titles using regex
    try:
        print(f"[INFO] Attempting regex extraction of movie titles...")
        
        # Find all movie title patterns - handle both "title" and 'title'
        title_pattern = r'["\']title["\']\s*:\s*["\']([^"\']+)["\']'
        year_pattern = r'["\']year["\']\s*:\s*(\d{4})'
        why_pattern = r'["\']why["\']\s*:\s*["\']([^"\']*)["\']'
        
        titles = re.findall(title_pattern, response_text)
        years = re.findall(year_pattern, response_text)
        whys = re.findall(why_pattern, response_text)
        
        print(f"[DEBUG] Regex found - titles: {titles}, years: {years}")
        
        if titles:
            recommendations = []
            for i, title in enumerate(titles):
                rec = {"title": title}
                if i < len(years):
                    rec["year"] = int(years[i])
                else:
                    rec["year"] = ""
                if i < len(whys) and whys[i]:
                    rec["why"] = whys[i]
                else:
                    rec["why"] = "Recommended based on your preferences"
                recommendations.append(rec)
            
            print(f"[INFO] Regex extraction found {len(recommendations)} movies: {[r['title'] for r in recommendations]}")
            return {"recommendations": recommendations, "user_mentioned_movies": []}
    except Exception as e:
        print(f"[WARN] Regex extraction failed: {e}")
    
    print(f"[ERROR] All parsing attempts failed")
    return None


def filter_mentioned_movies(rec: dict, user_message: str) -> dict:
    """Filter out any movies that the user mentioned from the recommendations."""
    if not rec:
        return rec
    
    recommendations = rec.get("recommendations", [])
    mentioned = rec.get("user_mentioned_movies", [])
    
    # Also extract potential movie mentions from the user message
    # Common patterns: "I like X", "I love X", "I watched X", "similar to X"
    user_message_lower = user_message.lower()
    
    # Add any movies from user_mentioned_movies to our filter list
    mentioned_lower = [m.lower() for m in mentioned]
    
    # Filter out recommendations that match mentioned movies
    filtered_recommendations = []
    for movie in recommendations:
        title = movie.get("title", "")
        title_lower = title.lower()
        
        # Check if this movie title appears in mentioned movies or user message
        is_mentioned = False
        
        # Check against extracted mentioned movies
        for mentioned_title in mentioned_lower:
            if mentioned_title in title_lower or title_lower in mentioned_title:
                is_mentioned = True
                print(f"[INFO] Filtering out '{title}' - matches mentioned movie '{mentioned_title}'")
                break
        
        # Also check if the title appears directly in user message
        if not is_mentioned and title_lower in user_message_lower:
            is_mentioned = True
            print(f"[INFO] Filtering out '{title}' - appears in user message")
        
        if not is_mentioned:
            filtered_recommendations.append(movie)
    
    if len(filtered_recommendations) < len(recommendations):
        print(f"[INFO] Filtered {len(recommendations) - len(filtered_recommendations)} mentioned movies from recommendations")
    
    rec["recommendations"] = filtered_recommendations
    return rec


def format_recommendation(rec: dict, user_message: str = ""):
    """Format the recommendations as a nice response with TMDB posters and ratings.
    
    Returns gr.HTML for rich content (with TMDB) or plain string (without TMDB).
    """
    if not rec:
        return "I couldn't generate proper recommendations. Please try again!"
    
    # Filter out any movies the user mentioned
    if user_message:
        rec = filter_mentioned_movies(rec, user_message)
    
    recommendations = rec.get("recommendations", [])
    
    if not recommendations:
        return "I couldn't generate proper recommendations. Please try again!"
    
    # Check if TMDB API is available
    has_tmdb = bool(get_tmdb_api_key())
    
    if has_tmdb:
        response = "<h2>Movie Recommendations</h2>\n"
        for i, movie in enumerate(recommendations, 1):
            title = movie.get("title", "Unknown")
            year = movie.get("year", "")
            why = movie.get("why", "")
            response += format_movie_card_with_tmdb(title, year, why, i)
        response += '<p style="font-size: 0.8em; color: #888;">Movie data from TMDB</p>'
        # Return gr.HTML component for rich HTML rendering in chatbot
        return gr.HTML(response)
    else:
        # Fallback to text-only format if no TMDB key
        response = "Here are my movie recommendations for you:\n\n"
        for i, movie in enumerate(recommendations, 1):
            title = movie.get("title", "Unknown")
            year = movie.get("year", "")
            why = movie.get("why", "")
            response += f"**{i}. {title}** ({year})\n"
            response += f"   {why}\n\n"
    
    return response


# ============================================================================
# LOCAL MODEL INFERENCE
# ============================================================================
def run_local_model(messages: list[dict], max_tokens: int, temperature: float) -> str:
    """Run inference using the local transformers pipeline."""
    try:
        print(f"[INFO] Starting local model inference...")
        print(f"[INFO] Parameters: max_tokens={max_tokens}, temperature={temperature}")
        print(f"[INFO] Number of messages in context: {len(messages)}")
        
        print(f"[INFO] Running inference on {LOCAL_MODEL_NAME}...")
        result = local_pipeline(
            messages,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=max(temperature, 0.01),  # Avoid temperature=0 issues
        )
        print(f"[INFO] Inference complete, extracting response...")
        
        # Extract the assistant's response from the pipeline output
        raw_content = result[0]["generated_text"][-1]["content"]
        print(f"[INFO] Response generated ({len(raw_content)} characters)")
        return raw_content
    except Exception as e:
        print(f"[ERROR] Local model inference failed: {e}")
        return f"Error generating response: {str(e)}"


def run_local_model_streaming(messages: list[dict], max_tokens: int, temperature: float):
    """Run inference using the local transformers pipeline with simulated streaming."""
    try:
        print(f"[INFO] Starting local model streaming inference...")
        print(f"[INFO] Parameters: max_tokens={max_tokens}, temperature={temperature}")
        
        # Local pipeline doesn't support true streaming, so we run and yield in chunks
        print(f"[INFO] Running inference on {LOCAL_MODEL_NAME}...")
        result = local_pipeline(
            messages,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=max(temperature, 0.01),
        )
        print(f"[INFO] Inference complete, streaming response...")
        
        raw_content = result[0]["generated_text"][-1]["content"]
        print(f"[INFO] Response generated ({len(raw_content)} characters), streaming to UI...")
        
        # Simulate streaming by yielding character by character (or in chunks)
        chunk_size = 5
        for i in range(0, len(raw_content), chunk_size):
            yield raw_content[:i + chunk_size]
        print(f"[INFO] Streaming complete")
    except Exception as e:
        print(f"[ERROR] Local model streaming failed: {e}")
        yield f"Error generating response: {str(e)}"


# ============================================================================
# API MODEL INFERENCE
# ============================================================================
def run_api_model(messages: list[dict], max_tokens: int, temperature: float, top_p: float, hf_token: str):
    """Run inference using the HuggingFace Inference API."""
    print(f"[INFO] Starting API model inference...")
    print(f"[INFO] Model: {API_MODEL_NAME}")
    print(f"[INFO] Parameters: max_tokens={max_tokens}, temperature={temperature}, top_p={top_p}")
    
    client = InferenceClient(token=hf_token, model=API_MODEL_NAME)
    
    print(f"[INFO] Sending request to HuggingFace API...")
    response = ""
    chunk_count = 0
    for chunk in client.chat_completion(
        messages,
        max_tokens=max_tokens,
        stream=True,
        temperature=temperature,
        top_p=top_p,
    ):
        if chunk.choices and chunk.choices[0].delta.content:
            response += chunk.choices[0].delta.content
            chunk_count += 1
            yield response
    print(f"[INFO] API response complete ({len(response)} characters, {chunk_count} chunks)")


# ============================================================================
# CHAT RESPONSE FUNCTION
# ============================================================================
if LOCAL_MODEL:
    def respond(
        message,
        history,
        system_message,
        max_tokens,
        temperature,
        top_p,
        genre,
        mood,
        era,
        viewing_pref,
        pace,
    ):
        print(f"\n{'='*60}")
        print(f"[INFO] New chat request received (LOCAL MODE)")
        print(f"[INFO] User message: {message[:100]}..." if len(message) > 100 else f"[INFO] User message: {message}")
        print(f"[INFO] Preferences - Genre: {genre}, Mood: {mood}, Era: {era}, Viewing: {viewing_pref}, Pace: {pace}")
        
        token = None  # Not needed for local model
        
        # Build user preferences
        print(f"[INFO] Step 1: Building user preferences...")
        user_answers = {
            "mood": mood or "Any",
            "genres": [genre] if genre else ["Any"],
            "pace": pace or "Balanced",
            "viewing_context": viewing_pref or "Any",
            "era": era or "Any Era",
            "open_ended": message
        }
        
        # Check if user is asking for recommendations
        recommendation_keywords = ["recommend", "suggest", "movie", "watch", "looking for", "i like", "want to see", "what should", "recommendations"]
        is_recommendation_request = any(kw in message.lower() for kw in recommendation_keywords)
        print(f"[INFO] Step 2: Detected recommendation request: {is_recommendation_request}")
        print(f"[INFO] All preferences set: {all([genre, mood, era, viewing_pref])}")
        
        if is_recommendation_request and all([genre, mood, era, viewing_pref]):
            print(f"[INFO] Step 3: Using STRUCTURED recommendation mode")
            # Use structured recommendation approach
            print(f"[INFO] Step 4: Building few-shot prompt with examples...")
            messages = build_messages(user_answers)
            print(f"[INFO] Prompt built with {len(messages)} messages")
            
            # Local model inference
            print(f"[INFO] Step 5: Running LLM inference...")
            response = run_local_model(messages, max_tokens, temperature)
            
            print(f"[INFO] Step 6: Cleaning model output...")
            cleaned = clean_output(response)
            
            print(f"[INFO] Step 7: Parsing JSON recommendation...")
            rec = parse_recommendation(cleaned)
            
            if rec:
                print(f"[INFO] Step 8: Successfully parsed {len(rec.get('recommendations', []))} recommendations")
                print(f"[INFO] Step 9: Filtering mentioned movies and fetching TMDB data...")
                formatted = format_recommendation(rec, message)
                print(f"[INFO] Step 10: Response ready, sending to UI")
                yield formatted
            else:
                print(f"[WARN] Step 8: Failed to parse JSON, returning raw response")
                yield f"Here's my recommendation based on your preferences:\n\n{cleaned}"
        else:
            print(f"[INFO] Step 3: Using CONVERSATIONAL mode")
            # General conversation mode with preferences context
            preferences_context = f"""You are a knowledgeable and friendly movie recommendation assistant.

User's Movie Preferences:
- Favorite Genre: {genre or 'Not specified'}
- Preferred Mood: {mood or 'Not specified'}
- Era Preference: {era or 'Not specified'}
- Viewing Context: {viewing_pref or 'Not specified'}
- Preferred Pace: {pace or 'Not specified'}

Your task:
1. If they ask for recommendations, suggest 3-5 specific movie titles with brief descriptions
2. Explain why each movie fits their preferences
3. Be enthusiastic and conversational
4. If they ask follow-up questions, continue to tailor recommendations to their stated preferences
5. If preferences are not fully specified, ask clarifying questions"""

            print(f"[INFO] Step 4: Building conversation context...")
            messages = [{"role": "system", "content": preferences_context}]
            messages.extend(history)
            messages.append({"role": "user", "content": message})
            print(f"[INFO] Context built with {len(messages)} messages (including {len(history)} history)")

            # Local model with simulated streaming
            print(f"[INFO] Step 5: Running LLM inference with streaming...")
            for partial in run_local_model_streaming(messages, max_tokens, temperature):
                yield partial
        
        print(f"[INFO] Request complete")
        print(f"{'='*60}\n")
else:
    def respond(
        message,
        history,
        system_message,
        max_tokens,
        temperature,
        top_p,
        hf_token: gr.OAuthToken | None,
        genre,
        mood,
        era,
        viewing_pref,
        pace,
    ):
        print(f"\n{'='*60}")
        print(f"[INFO] New chat request received (API MODE)")
        print(f"[INFO] User message: {message[:100]}..." if len(message) > 100 else f"[INFO] User message: {message}")
        print(f"[INFO] Preferences - Genre: {genre}, Mood: {mood}, Era: {era}, Viewing: {viewing_pref}, Pace: {pace}")
        
        # Check for authentication
        if hf_token is None or not getattr(hf_token, "token", None):
            print(f"[WARN] No HuggingFace token provided")
            yield "Please log in with your Hugging Face account first."
            return
        token = hf_token.token
        print(f"[INFO] HuggingFace token verified")
        
        # Build user preferences
        print(f"[INFO] Step 1: Building user preferences...")
        user_answers = {
            "mood": mood or "Any",
            "genres": [genre] if genre else ["Any"],
            "pace": pace or "Balanced",
            "viewing_context": viewing_pref or "Any",
            "era": era or "Any Era",
            "open_ended": message
        }
        
        # Check if user is asking for recommendations
        recommendation_keywords = ["recommend", "suggest", "movie", "watch", "looking for", "want to see", "what should"]
        is_recommendation_request = any(kw in message.lower() for kw in recommendation_keywords)
        print(f"[INFO] Step 2: Detected recommendation request: {is_recommendation_request}")
        print(f"[INFO] All preferences set: {all([genre, mood, era, viewing_pref])}")
        
        if is_recommendation_request and all([genre, mood, era, viewing_pref]):
            print(f"[INFO] Step 3: Using STRUCTURED recommendation mode")
            # Use structured recommendation approach
            print(f"[INFO] Step 4: Building few-shot prompt with examples...")
            messages = build_messages(user_answers)
            print(f"[INFO] Prompt built with {len(messages)} messages")
            
            # API model inference
            print(f"[INFO] Step 5: Running LLM inference via API...")
            response = ""
            for partial in run_api_model(messages, max_tokens, 0.3, top_p, token):
                response = partial
            
            print(f"[INFO] Step 6: Cleaning model output...")
            cleaned = clean_output(response)
            
            print(f"[INFO] Step 7: Parsing JSON recommendation...")
            rec = parse_recommendation(cleaned)
            
            if rec:
                print(f"[INFO] Step 8: Successfully parsed {len(rec.get('recommendations', []))} recommendations")
                print(f"[INFO] Step 9: Filtering mentioned movies and fetching TMDB data...")
                formatted = format_recommendation(rec, message)
                print(f"[INFO] Step 10: Response ready, sending to UI")
                yield formatted
            else:
                print(f"[WARN] Step 8: Failed to parse JSON, returning raw response")
                yield f"Here's my recommendation based on your preferences:\n\n{cleaned}"
        else:
            print(f"[INFO] Step 3: Using CONVERSATIONAL mode")
            # General conversation mode with preferences context
            preferences_context = f"""You are a knowledgeable and friendly movie recommendation assistant.

User's Movie Preferences:
- Favorite Genre: {genre or 'Not specified'}
- Preferred Mood: {mood or 'Not specified'}
- Era Preference: {era or 'Not specified'}
- Viewing Context: {viewing_pref or 'Not specified'}
- Preferred Pace: {pace or 'Not specified'}

Your task:
1. If they ask for recommendations, suggest 3-5 specific movie titles with brief descriptions
2. Explain why each movie fits their preferences
3. Be enthusiastic and conversational
4. If they ask follow-up questions, continue to tailor recommendations to their stated preferences
5. If preferences are not fully specified, ask clarifying questions"""

            print(f"[INFO] Step 4: Building conversation context...")
            messages = [{"role": "system", "content": preferences_context}]
            messages.extend(history)
            messages.append({"role": "user", "content": message})
            print(f"[INFO] Context built with {len(messages)} messages (including {len(history)} history)")

            # API model with real streaming
            print(f"[INFO] Step 5: Running LLM inference with streaming...")
            for partial in run_api_model(messages, max_tokens, temperature, top_p, token):
                yield partial
        
        print(f"[INFO] Request complete")
        print(f"{'='*60}\n")


# ============================================================================
# GRADIO UI
# ============================================================================

# Build additional inputs for ChatInterface
# Note: State components will be created inside Blocks and passed via render
if LOCAL_MODEL:
    # Local mode: no OAuth token needed
    additional_inputs = [
        gr.Textbox(value="You are a friendly movie recommendation chatbot.", label="System message", visible=False),
        gr.Slider(1, 2048, 512, step=1, label="Max new tokens", visible=False),
        gr.Slider(0.1, 4.0, 0.3, step=0.1, label="Temperature", visible=False),
        gr.Slider(0.1, 1.0, 0.95, step=0.05, label="Top-p", visible=False),
        gr.State(None),  # g_state - genre
        gr.State(None),  # m_state - mood
        gr.State(None),  # e_state - era
        gr.State(None),  # v_state - viewing
        gr.State(None),  # p_state - pace
    ]
else:
    # API mode: OAuth token is automatically injected by Gradio
    additional_inputs = [
        gr.Textbox(value="You are a friendly movie recommendation chatbot.", label="System message", visible=False),
        gr.Slider(1, 2048, 512, step=1, label="Max new tokens", visible=False),
        gr.Slider(0.1, 4.0, 0.3, step=0.1, label="Temperature", visible=False),
        gr.Slider(0.1, 1.0, 0.95, step=0.05, label="Top-p", visible=False),
        gr.State(None),  # g_state - genre
        gr.State(None),  # m_state - mood
        gr.State(None),  # e_state - era
        gr.State(None),  # v_state - viewing
        gr.State(None),  # p_state - pace
    ]

# Define ChatInterface outside Blocks (following professor's pattern)
# Note: 'type' parameter removed in Gradio 6.x - messages format is now default
chatbot = gr.ChatInterface(
    fn=respond,
    additional_inputs=additional_inputs,
)

with gr.Blocks(css=custom_css) as demo:
    with gr.Row():
        gr.Markdown("<h1>MOVIE RECOMMENDATION CHATBOT</h1>")
        if not LOCAL_MODEL:
            gr.LoginButton()
    
    # Show mode indicator
    mode_text = "Local Model" if LOCAL_MODEL else "HuggingFace API"
    gr.Markdown(f"_Running in {mode_text} mode_")

    # Preference Settings panel (above chatbot)
    with gr.Accordion("Preference Settings", open=True):
        gr.Markdown("*Set your movie preferences to get personalized recommendations*")
        
        with gr.Row():
            with gr.Column():
                # Genre selection
                g_radio = gr.Radio(
                    choices=GENRES,
                    label="What is your favourite genre?",
                    interactive=True
                )
                g_status = gr.Markdown()

                # Mood selection
                m_radio = gr.Radio(
                    choices=MOODS,
                    label="What mood are you in?",
                    interactive=True
                )
                m_status = gr.Markdown()

            with gr.Column():
                # Era selection
                e_radio = gr.Radio(
                    choices=ERAS,
                    label="Which era of movies do you prefer?",
                    interactive=True
                )
                e_status = gr.Markdown()

                # Viewing context selection
                v_radio = gr.Radio(
                    choices=VIEWING_CONTEXTS,
                    label="What is your viewing context?",
                    interactive=True
                )
                v_status = gr.Markdown()

            with gr.Column():
                # Pace selection
                p_radio = gr.Radio(
                    choices=PACE_OPTIONS,
                    label="Do you prefer fast-paced or slower movies?",
                    interactive=True
                )
                p_status = gr.Markdown()

    # Status indicator for preferences (between settings and chatbot)
    o_status = gr.Markdown("Set your preferences above, then start chatting to get recommendations!")

    # Render the chatbot interface
    chatbot.render()

    # Functions to handle user selections
    def set_genre(g):
        return g, f"Genre selected: *{g}*"

    def set_mood(m):
        return m, f"Mood selected: *{m}*"

    def set_era(e):
        return e, f"Era selected: *{e}*"

    def set_viewing_pref(v):
        return v, f"Viewing preference selected: *{v}*"

    def set_pace(p):
        return p, f"Pace selected: *{p}*"

    # Get references to the state components in additional_inputs
    g_state = additional_inputs[4]
    m_state = additional_inputs[5]
    e_state = additional_inputs[6]
    v_state = additional_inputs[7]
    p_state = additional_inputs[8]

    # Event handlers for each question - update the state components
    g_radio.change(
        fn=set_genre,
        inputs=g_radio,
        outputs=[g_state, g_status],
    )

    m_radio.change(
        fn=set_mood,
        inputs=m_radio,
        outputs=[m_state, m_status],
    )

    e_radio.change(
        fn=set_era,
        inputs=e_radio,
        outputs=[e_state, e_status],
    )

    v_radio.change(
        fn=set_viewing_pref,
        inputs=v_radio,
        outputs=[v_state, v_status],
    )

    p_radio.change(
        fn=set_pace,
        inputs=p_radio,
        outputs=[p_state, p_status],
    )


if __name__ == "__main__":
    demo.launch()
