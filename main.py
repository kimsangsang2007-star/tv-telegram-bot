from fastapi import FastAPI, HTTPException, Request
import requests

app = FastAPI()

# 💡 កុំភ្លេចប្តូរ Token និង Chat ID របស់អ្នកនៅទីនេះ
BOT_TOKEN = "8684494688:AAGc1zpMs_POn4Bd0J_Pb5UuVSQjJjfK_To"
CHAT_ID = "-1004327947082"
WEBHOOK_PASSPHRASE = "MY_SECRET_KEY_123"


@app.get("/")
def home():
    return {"status": "online", "message": "Bot is running!"}


@app.post("/webhook")
async def webhook(request: Request):
    try:
        data = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON")

    if data.get("passphrase") != WEBHOOK_PASSPHRASE:
        raise HTTPException(status_code=401, detail="Unauthorized")

    ticker = data.get("ticker", "UNKNOWN")
    action = data.get("action", "ALERT")
    price = data.get("price", "N/A")
    timeframe = data.get("timeframe", "")

    msg = (
        f"🚨 *TRADINGVIEW SIGNAL ALERT*\n\n"
        f"📈 *Asset:* `{ticker}` ({timeframe})\n"
        f"🎯 *Action:* `{action.upper()}`\n"
        f"💵 *Price:* `{price}`"
    )

    requests.post(
        f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
        json={
            "chat_id": TELEGRAM_CHAT_ID,
            "text": msg,
            "parse_mode": "Markdown",
        },
    )

    return {"status": "success"}
