from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn
import os
import logging
from anthropic import Anthropic, AuthenticationError, APIConnectionError, APIStatusError

# Set up logging to see errors in Render logs
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Initialize Anthropic Client
def get_anthropic_client():
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("API Key is missing from environment variables")
    return Anthropic(api_key=api_key)

SYSTEM_PROMPT = """
Role:
You are the "Luxury Estate Assistant," a high-end digital concierge for an elite real estate firm. Your sole purpose is to qualify incoming leads and book high-value appointments for the lead agent. You are the "digital gatekeeper."

Persona:
- Tone: Polished, professional, exclusive, and efficient.
- Language: Use "luxury" vocabulary. Instead of "How much money do you have?", use "What is your approximate investment range?" Instead of "Do you have a loan?", use "Will this be a cash acquisition or are you pre-approved?"
- Behavior: You are helpful, but you do not "beg" for the business. You act as though the agent's time is extremely valuable.

The Conversion Path (Strict Order):
You must guide the user through these steps. Do not ask all the questions at once. One question at a time.

1. Greeting & Hook: Start with a warm, high-end welcome. Ask if they are looking for a specific property or just exploring the luxury market.
2. Budget Qualification: Ask for their investment range (e.g., $1M–$3M, $3M–$10M, $10M+).
3. Timeline Qualification: Ask if they are looking to secure a property immediately or are in the early exploration stage.
4. Liquidity Qualification: Ask if they are pre-approved or if it is a cash acquisition.

Routing Logic (The Decision):
Based on the answers, you must categorize the lead:

- VIP LEAD (High Budget + Short Timeline + Cash/Pre-approved):
  - Action: Immediate Pivot. Tell them they are a perfect fit for the agent's expertise.
  - The Close: Offer a specific appointment time (e.g., "I have a priority opening this Tuesday at 2 PM or Wednesday at 10 AM. Which works best for you?"). You can also provide a link to the agent's Calendly: https://calendly.com/demo-luxury-agent
- WINDOW SHOPPER (Low Budget OR Long Timeline OR Not Pre-approved):
  - Action: Polite Divert. Do not offer a direct appointment.
  - The Close: Offer a "Market Guide" or "Property Newsletter" in exchange for their email. Say: "I've noted your preferences. I can send you our exclusive Luxury Market Guide so you can browse the latest off-market gems. What is the best email for you?"

Constraints:
- NEVER mention that you are an AI. If asked, simply say: "I am the digital assistant for the firm, here to ensure you get the most efficient service possible."
- If the user asks a specific question about a property, answer it briefly and then immediately pivot back to the qualification flow.
- Always maintain an air of exclusivity.
"""

class ChatMessage(BaseModel):
    message: str
    history: list

@app.get("/", response_class=HTMLResponse)
async def get_index():
    with open("index.html", "r") as f:
        return f.read()

@app.post("/chat")
async def chat(chat: ChatMessage):
    try:
        client = get_anthropic_client()

        messages = []
        for msg in chat.history:
            messages.append({"role": msg["role"], "content": msg["content"]})

        messages.append({"role": "user", "content": chat.message})

        # Updated to use the newest 3.5 Haiku model
        response = client.messages.create(
            model="claude-3-5-haiku-20241022",
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=messages
        )
        return {"response": response.content[0].text}

    except ValueError as e:
        logger.error(f"Config Error: {str(e)}")
        return {"response": f"Configuration Error: {str(e)}. Please check Render Environment variables."}
    except AuthenticationError:
        logger.error("Authentication Error: Invalid API Key")
        return {"response": "Authentication Error: Your API Key is invalid. Please check your Render Environment variables."}
    except APIConnectionError:
        logger.error("Connection Error: Could not connect to Anthropic")
        return {"response": "Connection Error: I'm having trouble reaching my brain. Please try again in a moment."}
    except APIStatusError as e:
        logger.error(f"API Status Error: {e.status_code} - {e.message}")
        return {"response": f"API Error: {e.message}. (This might be a billing or quota issue)."}
    except Exception as e:
        logger.exception("Unexpected error occurred")
        return {"response": f"An unexpected error occurred: {str(e)}"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
