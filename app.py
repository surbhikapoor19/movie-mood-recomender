import gradio as gr
from huggingface_hub import InferenceClient

GENRES = ["Horror", "Action", "Thriller", "Comedy", "Science-Fiction", "Drama", "Documentary", "Romance", "Animation"]
MOODS = ["Dark & Intense", "Light & Fun", "Emotional & Deep", "Suspenseful", "Inspirational"]
ERAS = ["Classic", "90s Classics", "2000s", "2010s", "Recent (2020+)", "Any Era"]
VIEWING_PREFS = ["Solo viewing", "Family-friendly", "With friends", "Any"]

def respond(
    message,
    history,
    system_message,
    max_tokens,
    temperature,
    top_p,
    hf_token: gr.OAuthToken,
    genre,
    mood,
    era,
    viewing_pref,
):
    preferences_context = f"""
You are a knowledgeable and friendly movie recommendation assistant.
User's Movie Preferences:
- Favorite Genre: {genre}
- Preferred Mood: {mood}
- Era Preference: {era}
- Viewing Context: {viewing_pref}
Your task:
1. Recommend movies that match these preferences
2. Provide 3-5 specific movie titles with brief descriptions
3. Explain why each movie fits their preferences
4. Be enthusiastic and conversational
5. If they ask follow-up questions, continue to tailor recommendations to their stated preferences
"""
    
    system_message = preferences_context
      
    print("[MODE] api")

    if hf_token is None or not getattr(hf_token, "token", None):
        yield "⚠️ Please log in with your Hugging Face account first."
        return
    
    client = InferenceClient(token=hf_token.token, model="openai/gpt-oss-20b")

    messages = [{"role": "system", "content": system_message}]
    messages.extend(history)
    messages.append({"role": "user", "content": message})

    response = ""

    for chunk in client.chat_completion(
        messages,
        max_tokens=max_tokens,
        stream=True,
        temperature=temperature,
        top_p=top_p,
    ):
        if chunk.choices and chunk.choices[0].delta.content:
            response += chunk.choices[0].delta.content
            yield response


with gr.Blocks() as demo:
    gr.Markdown("<h1 style='text-align:center;'>🎥 AI Movie Recommendation System</h1>")

    gr.LoginButton()

    # State variables for user preferences
    genre_state = gr.State(None)
    mood_state = gr.State(None)
    era_state = gr.State(None)
    viewing_pref_state = gr.State(None)
    chat_history = gr.State([])

    gr.Markdown("📽️What's your favorite movie genre?")
    genre_radio = gr.Radio(
        choices=GENRES,
        label="Select Genre",
        interactive=True
    )
    genre_status = gr.Markdown("❗ Please select a genre")
    
    gr.Markdown("🎭What mood are you in?")
    mood_radio = gr.Radio(
        choices=MOODS,
        label="Select Mood",
        interactive=True
    )
    mood_status = gr.Markdown("❗ Please select a mood")

    gr.Markdown("📅Which era of movies do you prefer?")
    era_radio = gr.Radio(
        choices=ERAS,
        label="Select Era",
        interactive=True
    )
    era_status = gr.Markdown("❗ Please select an era")

    gr.Markdown("👥What's your viewing context?")
    viewing_pref_radio = gr.Radio(
        choices=VIEWING_PREFS,
        label="Select Viewing Preference",
        interactive=True
    )
    viewing_pref_status = gr.Markdown("❗ Please select a viewing preference")

    overall_status = gr.Markdown("⏳ Complete all 4 questions to start chatting!")

    chatbot = gr.ChatInterface(
        respond,
        type="messages",
        additional_inputs=[
            gr.Textbox(value="You are a friendly movie recommendation chatbot.", label="System message"),
            gr.Slider(1, 2048, 512, step=1, label="Max new tokens"),
            gr.Slider(0.1, 4.0, 0.7, step=0.1, label="Temperature"),
            gr.Slider(0.1, 1.0, 0.95, step=0.05, label="Top-p"),
            genre_state,
            mood_state,
            era_state,
            viewing_pref_state,
        ],
        chatbot=gr.Chatbot(type="messages", value=[]),
    )

    # Functions to handle user selections
    def set_genre(g):
        return g, f"✅ Genre selected: *{g}*"

    def set_mood(m):
        return m, f"✅ Mood selected: *{m}*"

    def set_era(e):
        return e, f"✅ Era selected: *{e}*"

    def set_viewing_pref(v):
        return v, f"✅ Viewing preference selected: *{v}*"

    def check_all_selections(genre, mood, era, viewing_pref):
        """Check if all questions are answered and generate initial message"""
        if all([genre, mood, era, viewing_pref]):
            initial_message = [
                {
                    "role": "assistant",
                    "content": f"""🎬 Perfect!"""
                }
            ]
            status = "✅ All set! Start chatting to get movie recommendations!"
            return initial_message, status
        else:
            missing = []
            if not genre: missing.append("Genre")
            if not mood: missing.append("Mood")
            if not era: missing.append("Era")
            if not viewing_pref: missing.append("Viewing Preference")
            
            status = f"⏳ Please complete: {', '.join(missing)}"
            return [], status

    # Event handlers for each question
    genre_radio.change(
        fn=set_genre,
        inputs=genre_radio,
        outputs=[genre_state, genre_status],
    ).then(
        fn=check_all_selections,
        inputs=[genre_state, mood_state, era_state, viewing_pref_state],
        outputs=[chat_history, overall_status]
    )

    mood_radio.change(
        fn=set_mood,
        inputs=mood_radio,
        outputs=[mood_state, mood_status],
    ).then(
        fn=check_all_selections,
        inputs=[genre_state, mood_state, era_state, viewing_pref_state],
        outputs=[chat_history, overall_status]
    )

    era_radio.change(
        fn=set_era,
        inputs=era_radio,
        outputs=[era_state, era_status],
    ).then(
        fn=check_all_selections,
        inputs=[genre_state, mood_state, era_state, viewing_pref_state],
        outputs=[chat_history, overall_status]
    )

    viewing_pref_radio.change(
        fn=set_viewing_pref,
        inputs=viewing_pref_radio,
        outputs=[viewing_pref_state, viewing_pref_status],
    ).then(
        fn=check_all_selections,
        inputs=[genre_state, mood_state, era_state, viewing_pref_state],
        outputs=[chat_history, overall_status]
    )

    # Update chatbot with initial message
    def update_chat(chat_messages):
        return chat_messages
    
    chat_history.change(update_chat, chat_history, chatbot.chatbot)

    chatbot.render()


if __name__== "__main__":
    demo.launch()