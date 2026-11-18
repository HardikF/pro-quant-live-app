import json
import os
import urllib.request


def main():
    print("=========================================")
    print("  ProQuant Telegram Alert Setup (2 Min)")
    print("=========================================")
    print("1. Open Telegram and search for @BotFather")
    print("2. Send /newbot, pick a name, and copy the HTTP API Token.")
    token = input("\nPaste your Telegram Bot Token here: ").strip()
    if not token:
        print("No token entered. Exiting.")
        return

    print("\n3. Now open your new bot in Telegram, click START, and send any message (like 'hi').")
    input("Press Enter AFTER you have sent 'hi' to your bot...")

    url = f"https://api.telegram.org/bot{token}/getUpdates"
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            data = json.loads(resp.read().decode())
        results = data.get("result", [])
        if not results:
            print("[ERROR] No messages found yet. Make sure you clicked START and sent 'hi' to your bot, then run this again.")
            return
        chat_id = str(results[-1]["message"]["chat"]["id"])
    except Exception as exc:
        print(f"[ERROR] Could not connect to Telegram: {exc}")
        return

    env_path = ".env"
    lines = []
    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

    new_lines = [l for l in lines if not l.startswith("TELEGRAM_BOT_TOKEN=") and not l.startswith("TELEGRAM_CHAT_ID=")]
    new_lines.append(f"TELEGRAM_BOT_TOKEN={token}\n")
    new_lines.append(f"TELEGRAM_CHAT_ID={chat_id}\n")
    with open(env_path, "w", encoding="utf-8") as f:
        f.writelines(new_lines)

    test_url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = json.dumps({"chat_id": chat_id, "text": "ProQuant Connected! You will now receive stock calls here automatically."}).encode()
    req = urllib.request.Request(test_url, data=payload, headers={"Content-Type": "application/json"})
    urllib.request.urlopen(req, timeout=10)
    print(f"\n[SUCCESS] Connected! Chat ID {chat_id} saved to .env and a test message was sent to your Telegram.")


if __name__ == "__main__":
    main()
