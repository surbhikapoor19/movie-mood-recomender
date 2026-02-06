import os
import json
import gradio as gr
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

load_dotenv()


# ── Prompt engineering ────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "DO NOT RECCOMEND A MOVIE IN user_mentioned_movies.\n"
    "You are a movie recommender. Your job: reccomend exactly ONE movie.\n\n"
    "Output schema:\n"
    '{"why":"<1 short sentence>","user_mentioned_movies":["<title>"],'
    '"recommended_movie":"<title>","year":<year>}'
)

FEW_SHOT_EXAMPLES = [
    {
        "user": {
            "mood": "scary",
            "genres": ["horror", "thriller"],
            "pace": "fast",
            "spectacle_vs_story": "story",
            "familiar_vs_new": "new",
            "open_ended": "Recently I watched a cool movie called Skinamarink. "
                          "I like tense movies with clever twists, not gore.",
        },
        "assistant": {
            "why": "Fast, tense, clever suspense with minimal gore.",
            "user_mentioned_movies": ["Skinamarink"],
            "recommended_movie": "A Quiet Place",
            "year": 2018,
        },
    },
    {
        "user": {
            "mood": "emotional",
            "genres": ["drama"],
            "pace": "slow",
            "spectacle_vs_story": "story",
            "familiar_vs_new": "familiar",
            "open_ended": "I love character growth and bittersweet endings. "
                          "Kinda like the Shawshank Redemption.",
        },
        "assistant": {
            "why": "Character-driven drama with emotional growth and warmth.",
            "user_mentioned_movies": ["Shawshank Redemption"],
            "recommended_movie": "Good Will Hunting",
            "year": 1997,
        },
    },
]

USER_PROMPT_TEMPLATE = (
    "Given the user's answers below, pick ONE movie recommendation.\n\n"
    "User answers (JSON):\n"
    "{answers_json}\n\n"
    "If the user mentions a movie, pick one similar to the movie they mention."
)


# ── Message builder ───────────────────────────────────────────────────

def build_messages(user_answers: dict) -> list[dict]:
    """Assemble system prompt, few-shot examples, and the user query."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    for ex in FEW_SHOT_EXAMPLES:
        messages.append({
            "role": "user",
            "content": USER_PROMPT_TEMPLATE.format(
                answers_json=json.dumps(ex["user"])
            ),
        })
        messages.append({
            "role": "assistant",
            "content": json.dumps(ex["assistant"]),
        })

    messages.append({
        "role": "user",
        "content": USER_PROMPT_TEMPLATE.format(
            answers_json=json.dumps(user_answers)
        ),
    })
    return messages


# ── Core recommendation logic ────────────────────────────────────────

def call_llm(messages: list[dict], token: str) -> str:
    """Send messages to the HF Inference API and return the raw response."""
    client = InferenceClient(
        provider="together",
        token=token,
        model="Qwen/Qwen2.5-7B-Instruct",
    )
    result = client.chat_completion(
        messages,
        max_tokens=256,
        temperature=0.02,
        top_p=0.95,
    )
    return result.choices[0].message.content


def parse_recommendation(raw: str) -> dict | None:
    """Parse the JSON model output. Returns the dict or None on failure."""
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None


def format_recommendation(data: dict) -> str:
    """Turn a parsed recommendation dict into user-facing markdown."""
    title = data.get("recommended_movie", "Unknown")
    year = data.get("year", "")
    why = data.get("why", "")
    mentioned = data.get("user_mentioned_movies", [])

    output = f"## {title} ({year})\n\n"
    output += f"**Why:** {why}\n\n"
    if mentioned:
        output += f"*Based on your mention of: {', '.join(mentioned)}*"
    return output


def recommend_movie(
    mood, genres, pace, spectacle_vs_story, familiar_vs_new, open_ended,
    hf_token: gr.OAuthToken = None,
):
    if not genres:
        raise gr.Error("Please select at least one genre.")

    # Use OAuth token if available, otherwise fall back to .env
    token = None
    if hf_token is not None:
        token = hf_token.token
    else:
        token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

    if not token:
        raise gr.Error("No HuggingFace token found. Please log in or set HUGGINGFACEHUB_API_TOKEN in .env")

    user_answers = {
        "mood": mood,
        "genres": genres,
        "pace": pace,
        "spectacle_vs_story": spectacle_vs_story,
        "familiar_vs_new": familiar_vs_new,
        "open_ended": open_ended,
    }

    messages = build_messages(user_answers)
    raw = call_llm(messages, token)
    data = parse_recommendation(raw)

    if data is not None:
        return format_recommendation(data)
    return f"**Model response:**\n\n{raw}"


# ── Gradio UI ─────────────────────────────────────────────────────────

GENRE_CHOICES = [
    "Action", "Adventure", "Animation", "Comedy", "Crime",
    "Documentary", "Drama", "Fantasy", "Horror", "Musical",
    "Mystery", "Romance", "Sci-Fi", "Thriller", "Western",
]


def create_demo():
    with gr.Blocks(title="Movie Mood Recommender") as demo:
        gr.Markdown("# Movie Mood Recommender\n"
                    "Tell us how you're feeling and we'll find the perfect movie.")

        with gr.Sidebar():
            gr.LoginButton()

        with gr.Row():
            with gr.Column():
                mood = gr.Textbox(
                    label="What mood are you in?",
                    placeholder="e.g. happy, sad, adventurous, scary, chill…",
                )
                genres = gr.CheckboxGroup(
                    choices=GENRE_CHOICES,
                    label="Pick one or more genres",
                )
                pace = gr.Radio(
                    choices=["slow", "medium", "fast"],
                    label="Preferred pace",
                    value="medium",
                )
                spectacle_vs_story = gr.Radio(
                    choices=["spectacle", "balanced", "story"],
                    label="Spectacle vs. Story",
                    value="balanced",
                )
                familiar_vs_new = gr.Radio(
                    choices=["familiar", "no preference", "new"],
                    label="Something familiar or something new?",
                    value="no preference",
                )
                open_ended = gr.Textbox(
                    label="Anything else? (optional)",
                    placeholder="Mention movies you like, themes, actors, etc.",
                    lines=3,
                )
                submit_btn = gr.Button("Get Recommendation", variant="primary")

            with gr.Column():
                output = gr.Markdown(label="Your Recommendation")

        submit_btn.click(
            fn=recommend_movie,
            inputs=[mood, genres, pace, spectacle_vs_story, familiar_vs_new, open_ended],
            outputs=output,
        )
    return demo


if __name__ == "__main__":
    demo = create_demo()
    demo.launch()
