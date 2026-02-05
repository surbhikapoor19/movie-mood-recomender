import gradio as gr
from huggingface_hub import InferenceClient

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
    font-style=italic;
    font-weight:bold;
    font-size:40px;
    text-align:center;
    color:#663356;
    text-shadow: 2px 2px 4px #693256;
}
"""

G = ["Horror", "Action", "Thriller", "Comedy", "Science-Fiction", "Drama", "Documentary", "Romance", "Animation"]
M = ["Dark & Intense", "Light & Fun", "Emotional & Deep", "Suspenseful", "Inspirational"]
E = ["Classic", "90s Classics", "2000s", "2010s", "Recent", "Any Era"]
V = ["Solo", "Family", "friends", "Any"]

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
    
    messages = [{"role": "system", "content": system_message}]
    messages.extend(history)
    messages.append({"role": "user", "content": message})

    response = ""
    print("[MODE] api")

    if hf_token is None or not getattr(hf_token, "token", None):
        yield "⚠️ Please log in with your Hugging Face account first."
        return
    client = InferenceClient(token=hf_token.token, model="openai/gpt-oss-20b")        
    
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


with gr.Blocks(css=custom_css) as demo:
    gr.Markdown("<h1 style='text-align:center;'>🎥 MOVIE RECOMMENDATION CHATBOT</h1>")

    gr.LoginButton()

    # State variables for user preferences
    g_state = gr.State(None)
    m_state = gr.State(None)
    e_state = gr.State(None)
    v_state = gr.State(None)
    chat_history = gr.State([])

    gr.Markdown("📽️What is your favourite genre?")
    g_radio = gr.Radio(
        choices=G,
        label="Select the genre you like...",
        interactive=True
    )
    g_status = gr.Markdown()
    gr.Markdown("🎭what mood you are in? ")
    m_radio = gr.Radio(
        choices=M,
        label="Select your mood...",
        interactive=True
    )
    m_status = gr.Markdown()

    gr.Markdown("📅Which era of movies do you prefer?")
    e_radio = gr.Radio(
        choices=E,
        label="Select the timeline...",
        interactive=True
    )
    e_status = gr.Markdown()

    gr.Markdown("👥What is your viewing context?")
    v_radio = gr.Radio(
        choices=V,
        label="Select your Preference",
        interactive=True
    )
    v_status = gr.Markdown()

    o_status = gr.Markdown("⏳ Complete all 4 questions and start a chat to give the recommendations!")

    chatbot = gr.ChatInterface(
        respond,
        type="messages",
        additional_inputs=[
            gr.Textbox(value="You are a friendly movie recommendation chatbot.", label="System message"),
            gr.Slider(1, 2048, 512, step=1, label="Max new tokens"),
            gr.Slider(0.1, 4.0, 0.7, step=0.1, label="Temperature"),
            gr.Slider(0.1, 1.0, 0.95, step=0.05, label="Top-p"),
            g_state,
            m_state,
            e_state,
            v_state,
        ],
        chatbot=gr.Chatbot(type="messages", value=[]),
    )

    # Functions to handle user selections
    def set_genre(g):
        return g,f"✅ genre selected: *{g}*"

    def set_mood(m):
        return m,f"✅ mood selected: *{m}*"

    def set_era(e):
        return e,f"✅ era selected: *{e}*"

    def set_viewing_pref(v):
        return v,f"✅ view_preference selected: *{v}*"

    def check_all_selections(g, m, e, v):
        """Check if all questions are answered and generate initial message"""
        if all([g, m, e, v]):
            initial_message = [
                {
                    "role": "assistant",
                    "content": f"""🎬 Perfect! Please start a chat for recommendations"""
                }
            ]
            status = "✅ All set! Start chatting to get movie recommendations!"
            return initial_message, status
        else:
            missing = []
            if not g: missing.append("Genre")
            if not m: missing.append("Mood")
            if not e: missing.append("Era")
            if not v: missing.append("Viewing Preference")
            
            status = f"⏳ Please complete: {', '.join(missing)}"
            return [], status

    # Event handlers for each question
    g_radio.change(
        fn=set_genre,
        inputs=g_radio,
        outputs=[g_state, g_status],
    ).then(
        fn=check_all_selections,
        inputs=[g_state, m_state, e_state, v_state],
        outputs=[chat_history, o_status]
    )

    m_radio.change(
        fn=set_mood,
        inputs=m_radio,
        outputs=[m_state, m_status],
    ).then(
        fn=check_all_selections,
        inputs=[g_state, m_state, e_state, v_state],
        outputs=[chat_history, o_status]
    )

    e_radio.change(
        fn=set_era,
        inputs=e_radio,
        outputs=[e_state, e_status],
    ).then(
        fn=check_all_selections,
        inputs=[g_state, m_state, e_state, v_state],
        outputs=[chat_history, o_status]
    )

    v_radio.change(
        fn=set_viewing_pref,
        inputs=v_radio,
        outputs=[v_state, v_status],
    ).then(
        fn=check_all_selections,
        inputs=[g_state, m_state, e_state, v_state],
        outputs=[chat_history, o_status]
    )

    # Update chatbot with initial message
    def update_chat(chat_messages):
        return chat_messages
    
    chat_history.change(update_chat, chat_history, chatbot.chatbot)

    chatbot.render()


if __name__== "__main__":
    demo.launch()