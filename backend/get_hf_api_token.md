# How to Get a Hugging Face API Token

1. Go to [https://huggingface.co/join](https://huggingface.co/join) and create a free account (or log in if you already have one).

2. Once logged in, click on your profile icon (top right) and select **Settings**.

3. In the left sidebar, click on **Access Tokens**.

4. Click the **New token** button.

5. Give your token a name (e.g., "ghack") and select the **Read** role.

6. Click **Generate**.

7. Copy the generated token.

8. Paste the token into your `.env` file as:
   ```
   HF_API_TOKEN=your_token_here
   ```

9. Save the `.env` file and restart your FastAPI server.

You can now use the Hugging Face Inference API in your project!
