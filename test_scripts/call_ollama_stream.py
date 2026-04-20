from ollama import Client

client = Client(host="http://localhost:11434")

def ask_vlm_stream(model: str, prompt: str, image_paths: list[str]):
    print("starting request...")
    stream = client.chat(
        model=model,
        messages=[
            {
                "role": "user",
                "content": prompt,
                "images": image_paths,
            }
        ],
        stream=True,
    )

    full_content = []
    full_thinking = []

    for chunk in stream:
        msg = chunk.get("message", {})
        if msg.get("thinking"):
            print(msg["thinking"], end="", flush=True)
            full_thinking.append(msg["thinking"])
        if msg.get("content"):
            print(msg["content"], end="", flush=True)
            full_content.append(msg["content"])

    print("\n\nfinished")
    # print the final answer to a file
    with open("final_answer.txt", "w") as f:
        f.write("".join(full_content))


    return "".join(full_content)

print(ask_vlm_stream(
    model="gemini-3-flash-preview",
    prompt="Describe the important objects in this image. Who is winning in this position and why?",
    image_paths=["./puzzles/plain_boards/00008.png"],
))