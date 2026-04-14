from ollama import Client

client = Client(host="http://localhost:11434")

def ask_vlm(model: str, prompt: str, image_paths: list[str]):
    resp = client.chat(
        model=model,
        messages=[
            {
                "role": "user",
                "content": prompt,
                "images": image_paths,
            }
        ],
    )
    return resp["message"]["content"]

print(ask_vlm(
    model="gemini-3-flash-preview",
    prompt="Describe the important objects in this image. Who is winning in this position and why?",
    image_paths=["./puzzles/plain_boards/00008.png"],
))