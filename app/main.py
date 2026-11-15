from fastapi import FastAPI

from app.api.health import router as health_router

async def lifespan(app):
    app.state.ready=True
    yield
    app.state.ready=False


def create_app():
    app = FastAPI(title="FDE Sandbox API", version="0.1.0" ,lifespan=lifespan)
    app.include_router(health_router)
    return app

app = create_app()