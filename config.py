"""독립 실행 설정. 키는 로컬 .env 또는 Streamlit Secrets에서 읽는다."""
import os
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / '.env')
DATA = ROOT / 'data/documents.json'
RUNTIME = ROOT / '.runtime'
VECTOR_PATH = RUNTIME / 'chroma'
EMBEDDING_MODEL = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
GEMINI_BASE_URL = 'https://generativelanguage.googleapis.com/v1beta/openai/'
CHAT_MODEL = os.getenv('GEMINI_MODEL', 'gemini-3.5-flash-lite')
QUALITY_MODEL = CHAT_MODEL
CHUNK_SIZE = int(os.getenv('CHUNK_SIZE', '1000'))
CHUNK_OVERLAP = int(os.getenv('CHUNK_OVERLAP', '200'))
TOP_K = int(os.getenv('TOP_K', '6'))
CANDIDATE_K = int(os.getenv('CANDIDATE_K', '24'))
COLLECT_LIMIT = int(os.getenv('COLLECT_LIMIT', '6'))
if not 0 <= CHUNK_OVERLAP < CHUNK_SIZE or TOP_K < 1:
    raise ValueError('CHUNK_SIZE > CHUNK_OVERLAP >= 0, TOP_K >= 1이어야 합니다.')
if CANDIDATE_K < 1 or COLLECT_LIMIT < 1:
    raise ValueError('CANDIDATE_K와 COLLECT_LIMIT는 1 이상이어야 합니다.')
