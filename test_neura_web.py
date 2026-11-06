from neura.tools.web_fetch import WebFetch

print("=" * 60)
print("Neura Internet Connectivity Test")
print("=" * 60)

web = WebFetch()

try:
    result = web.fetch(
        "https://example.com"
    )

    print("Status:", result["status"])
    print("Content-Type:", result["content_type"])
    print("URL:", result["url"])

    print()
    print("First 300 characters:")
    print(result["body"][:300])

    print("=" * 60)

    if result["status"] == 200:
        print("RESULT: INTERNET ACCESS WORKING")
    else:
        print("RESULT: INTERNET ACCESS FAILED")

except Exception as e:
    print("=" * 60)
    print("RESULT: INTERNET ACCESS FAILED")
    print("Error:", type(e).__name__)
    print("Message:", e)

print("=" * 60)
