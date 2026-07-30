from fastapi import FastAPI, Depends, HTTPException
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel
from generate_result import answer_query
from auth import create_access_token, get_current_user, verify_password, FAKE_USERS
import time
import logging

# dict for query caching(normally it should be stored in db due to complexity we add in dict)
query_cache = {}
app = FastAPI(title="French Residency Assistant API")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Approximate cost per token for gpt-4o-mini (as of 2026)
COST_PER_INPUT_TOKEN = 0.00000015  # $0.15 per million
COST_PER_OUTPUT_TOKEN = 0.0000006  # $0.60 per million

# For Input Guardrail, Off-topic keywords that suggest non-residency questions
RESIDENCY_KEYWORDS = [
    "carte",
    "séjour",
    "résident",
    "visa",
    "ofii",
    "civic",
    "civique",
    "titre",
    "prefecture",
    "immigration",
    "naturalisation",
    "passeport",
    "permis",
    "residence",
    "residency",
    "french",
    "france",
    "foreigners",
    "cir",
    "vls",
    "nationality",
    "citizenship",
]


# For Output Guardrail, checks the source to verify groundness
def is_grounded(answer: str) -> bool:
    source_docs = [
        "carte_de_resident.txt",
        "carte_de_sejour_renewal.txt",
        "civic_exam.txt",
        "ofii_medical_visit.txt",
    ]

    return any(source in answer for source in source_docs)


# For Input Guardrail, to check question is residency related
def is_residency_related(question: str) -> bool:
    question_lower = question.lower()
    return any(keyword in question_lower for keyword in RESIDENCY_KEYWORDS)


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

    question = request.question.strip().lower()

    # Input guardrail 1
    if not question:
        return QueryResponse(
            answer="I can only answer questions about French residency documents, permits, and administrative processes.",
            question=request.question,
        )
    # Input guardrail 2
    if not is_residency_related(request.question):
        logger.info(f"Off-topic question rejected: '{question}'")
        return QueryResponse(
            answer="I can only answer questions about French residency documents, permits, and administrative processes.",
            question=request.question,
        )

    # checks query cache for repeated questions(entire 11-second pipeline bypassed for a repeated question.)
    if question in query_cache:
        logger.info(f"Cache hit for: {question}")
        return QueryResponse(answer=query_cache[question], question=request.question)

    # start time before call
    start_time = time.time()
    answer = answer_query(request.question)
    # calculate latency
    latency = time.time() - start_time

    # Output guardrail
    if not is_grounded(answer):
        logger.info(f"Ungrounded answer detected for: '{question}'")
        answer = (
            answer
            + "\n\n ⚠️ Note: This answer may not be fully grounded in the official documents."
        )

    # Rough token estimate (4 chars ≈ 1 token)
    input_tokens = len(request.question) / 4
    output_tokens = len(answer) / 4
    estimated_cost = (input_tokens * COST_PER_INPUT_TOKEN) + (
        output_tokens * COST_PER_OUTPUT_TOKEN
    )

    # log latency & cost
    logger.info(
        f"Cache miss | latency: {latency:.2f}s | estimated_cost: ${estimated_cost:.6f}"
    )
    # save cache
    query_cache[question] = answer
    return QueryResponse(answer=answer, question=request.question)
