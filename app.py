from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import torch
from transformers import pipeline
import os




app = FastAPI()

# Fix Hugging Face cache permission issue
os.environ["HF_HOME"] = "/tmp/huggingface"
os.environ["TRANSFORMERS_CACHE"] = "/tmp/huggingface"
os.environ["HF_DATASETS_CACHE"] = "/tmp/huggingface"
os.environ["HF_MODULES_CACHE"] = "/tmp/huggingface"


# Initialize the TinyLlama model pipeline once at startup
pipe = pipeline(
    "text-generation", 
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0", 
    torch_dtype=torch.bfloat16, 
    device_map="auto"
)

# Mount the static folder (for CSS & JS)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
def get_chat_page():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()

def generate_response(user_content: str) -> str:
    try:
        print(f"Generating response for user content: {user_content}")
        # Format messages using the tokenizer's chat template
        messages = [
            {
                "role": "system",
                "content": "You are a friendly chatbot who always talks by recognizing the user and their problem and then generates the best advice or solution for the user. and You are made by a person named Jazib Salman",
            },
            {
                "role": "user", 
                "content": user_content
            },
        ]
        
        # Apply chat template
        prompt = pipe.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        # Generate response
        outputs = pipe(
            prompt, 
            max_new_tokens=256, 
            do_sample=True, 
            temperature=0.7, 
            top_k=50, 
            top_p=0.95
        )
        
        # Extract the generated text
        generated_text = outputs[0]["generated_text"]
        
        # Remove the prompt part to get only the response
        response = generated_text[len(prompt):].strip()
        
        print(f"Generated response: {response}")
        return response
        
    except Exception as e:
        print(f"Error generating response: {str(e)}")
        raise e

@app.post("/chat", response_class=JSONResponse)
async def chat(request: Request):
    try:
        data = await request.json()
        content = data.get("content", "")
        
        if not content:
            return JSONResponse(
                content={"error": "No content provided"}, 
                status_code=400
            )
        
        # Generate response using TinyLlama
        response = generate_response(content)
        
        return JSONResponse(content={
            "response": response,
            "status": "success"
        })
        
    except Exception as e:
        print(f"Chat endpoint error: {str(e)}")
        return JSONResponse(
            content={"error": f"Failed to generate response: {str(e)}"}, 
            status_code=500
        )

if __name__ == "__main__":
    import uvicorn, os
    port = int(os.environ.get("PORT", 7860))  # HF passes PORT=7860
    uvicorn.run("app:app", host="0.0.0.0", port=port)
