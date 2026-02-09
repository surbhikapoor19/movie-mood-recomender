# Citations and References

Documentation sources referenced while building this Movie Recommendation Chatbot.

## Gradio Documentation

1. **Gradio 6 Migration Guide**
   - URL: https://www.gradio.app/main/guides/gradio-6-migration-guide
   - Used for: Understanding breaking changes between Gradio 5.x and 6.x

2. **Gradio ChatInterface - Creating a Chatbot Fast**
   - URL: https://www.gradio.app/main/guides/creating-a-chatbot-fast
   - Used for: `gr.ChatInterface` setup, `additional_inputs`, streaming responses, returning complex content like `gr.HTML`

3. **Gradio Chatbot Component**
   - URL: https://gradio.app/docs/gradio/chatbot
   - Used for: `sanitize_html`, `render_markdown` parameters, message format with `role` and `content`

4. **Gradio HTML Component**
   - URL: https://gradio.app/docs/gradio/html
   - Used for: Rendering HTML content in chatbot responses

## HuggingFace Documentation

5. **HuggingFace Inference Providers**
   - URL: https://huggingface.co/docs/inference-providers/en/faq
   - Used for: Understanding the provider-based routing system, available models, `InferenceClient` usage

6. **HuggingFace Inference Providers - Chat Completion**
   - URL: https://huggingface.co/docs/inference-providers/en/tasks/chat-completion
   - Used for: Chat completion API format, supported models like `openai/gpt-oss-120b`

## TMDB API

7. **TMDB API Documentation**
   - URL: https://developer.themoviedb.org/docs
   - Used for: Movie search endpoint, image base URLs (`https://image.tmdb.org/t/p/w300`)

## Python Libraries

8. **Transformers Pipeline**
   - URL: https://huggingface.co/docs/transformers/main_classes/pipelines
   - Used for: Local model inference with `pipeline("text-generation", ...)`, parameters like `max_new_tokens`, `temperature`, `do_sample`

9. **python-dotenv**
   - URL: https://pypi.org/project/python-dotenv/
   - Used for: Loading `.env` files for local development

10. **huggingface_hub InferenceClient**
    - URL: https://huggingface.co/docs/huggingface_hub/package_reference/inference_client
    - Used for: API-based model inference with streaming support
