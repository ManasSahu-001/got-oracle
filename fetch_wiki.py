import wikipediaapi

wiki = wikipediaapi.Wikipedia(language='en', user_agent='GOTChatBot/1.0')
pages = [
    "Jon Snow (character)",
    "Night King",
    "Battle of Winterfell",
    "Arya Stark",
    "Daenerys Targaryen",
    "Tyrion Lannister",
    "Cersei Lannister",
    "Jaime Lannister",
    "Sansa Stark",
    "Bran Stark",
]

with open("docs/got_wiki.txt", "w", encoding="utf-8") as f:
    for title in pages:
        page = wiki.page(title)
        if page.exists():
            print(f"✅ Fetched: {title}")
            f.write(f"=== {title} ===\n{page.text}\n\n")
        else:
            print(f"❌ Not found: {title}")

print("\nDone! Saved to docs/got_wiki.txt")


