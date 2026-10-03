import io

from fastapi import APIRouter, HTTPException, UploadFile, File
from pypdf import PdfReader

from app.core.rag import DocumentTooLargeError, ingest_document
from app.schemas.models import IngestRequest, IngestResponse

router = APIRouter()

SUPPORTED_MIME_TYPES = {
    "application/pdf",
    "text/plain",
    "text/markdown",
    "text/csv",
    "application/json",
    "text/html",
}


@router.post("/documents", response_model=IngestResponse)
async def ingest(request: IngestRequest) -> IngestResponse:
    if not request.text.strip():
        raise HTTPException(status_code=422, detail="Document text cannot be empty.")
    try:
        section_id = await ingest_document(request.text, source=request.source)
    except DocumentTooLargeError as exc:
        raise HTTPException(status_code=413, detail=str(exc)) from exc
    return IngestResponse(
        section_id=section_id,
        message=f"Document '{request.source}' ingested successfully.",
    )


@router.post("/documents/upload", response_model=IngestResponse)
async def upload_file(file: UploadFile = File(...)) -> IngestResponse:
    if file.content_type not in SUPPORTED_MIME_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{file.content_type}'. "
                   f"Supported types: PDF, plain text, Markdown, CSV, JSON, HTML.",
        )

    raw = await file.read()

    if file.content_type == "application/pdf":
        reader = PdfReader(io.BytesIO(raw))
        pages = [page.extract_text() or "" for page in reader.pages]
        text = "\n\n".join(pages).strip()
        if not text:
            raise HTTPException(
                status_code=422,
                detail="No text could be extracted from the PDF. "
                       "Scanned/image-only PDFs are not supported.",
            )
    else:
        text = raw.decode("utf-8", errors="replace").strip()
        if not text:
            raise HTTPException(status_code=422, detail="The uploaded file is empty.")

    source = file.filename or "uploaded"
    doc_id = await ingest_document(text, source=source)
    return IngestResponse(
        doc_id=doc_id,
        message=f"Document '{source}' ingested successfully.",
    )
