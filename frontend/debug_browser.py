import asyncio
from playwright.async_api import async_playwright

async def main():
    async with async_playwright() as p:
        try:
            browser = await p.chromium.launch()
            page = await browser.new_page()
            page.on('console', lambda msg: print(f'CONSOLE: {msg.type}: {msg.text}'))
            page.on('pageerror', lambda err: print(f'PAGE ERROR: {err}'))
            print("Navigating to localhost:5173...")
            await page.goto('http://localhost:5173', timeout=10000)
            await asyncio.sleep(2)
            await browser.close()
            print("Done.")
        except Exception as e:
            print("Script Error:", e)

if __name__ == "__main__":
    asyncio.run(main())
