import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

app = FastAPI(
    title="AI Analytics Pro",
    description="FastAPI wrapper for Vercel deployment.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "OPTIONS"],
    allow_headers=["*"],
)


@app.get("/", response_class=HTMLResponse)
async def root():
    return """
    <html>
      <head>
        <title>AI Analytics Pro</title>
      </head>
      <body style="font-family: Arial, sans-serif; background: #0f172a; color: #e2e8f0; margin: 0; padding: 40px;">
        <div style="max-width: 800px; margin: auto;">
          <h1 style="color: #60a5fa;">AI Analytics Pro</h1>
          <p>This repository uses Streamlit for its primary UI. Run the Streamlit app locally with:</p>
          <pre style="background: #111827; padding: 16px; border-radius: 8px; color: #f8fafc;">streamlit run streamlit_app.py</pre>
          <p>For Vercel deployment, this FastAPI wrapper exposes a compatible top-level <code>app</code> object.</p>
        </div>
      </body>
    </html>
    """


@app.get("/health")
async def health_check():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8000)),
        log_level="info",
    )
