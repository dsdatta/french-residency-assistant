from fastapi import FastAPI, Depends, HTTPException
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel
from generate_result import answer_query
from auth import create_access_token, get_current_user, verify_password, FAKE_USERS

app = FastAPI(title="French Residency Assistant API")


class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    question: str
    answer: str


class Token(BaseModel):
    access_token: str
    token_type: str


@app.get("/health")
async def health():
    return {"status": "OK"}


@app.post("/token", response_model=Token)
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = FAKE_USERS.get(form_data.username)
    if not user or not verify_password(form_data.password, user["hashed_password"]):
        raise HTTPException(status_code=400, detail="Incorrect username or password")

    access_token = create_access_token(data={"sub": form_data.username})
    return Token(access_token=access_token, token_type="bearer")


@app.post("/query", response_model=QueryResponse)
async def query_endpoint(
    request: QueryRequest, current_user: dict = Depends(get_current_user)
):
    answer = answer_query(request.question)
    return QueryResponse(answer=answer, question=request.question)
