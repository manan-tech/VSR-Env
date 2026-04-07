from vsr_env.server.app import app

def main():
    import uvicorn
    uvicorn.run("vsr_env.server.app:app", host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()
