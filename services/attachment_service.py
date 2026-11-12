"""Safe, bounded local uploads for chat attachments."""
import io
import re
import uuid
import zipfile
from pathlib import Path
from xml.etree import ElementTree

from werkzeug.utils import secure_filename


MAX_FILE_BYTES = 8 * 1024 * 1024
MAX_FILES = 4
MAX_EXTRACTED_CHARS = 12000
ALLOWED = {".png", ".jpg", ".jpeg", ".webp", ".pdf", ".docx", ".txt", ".csv"}


class AttachmentError(ValueError):
    pass


class AttachmentService:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def save(self, conversation_id, uploads):
        if len(uploads) > MAX_FILES:
            raise AttachmentError(f"Tối đa {MAX_FILES} tệp trong một tin nhắn.")
        prepared = []
        for upload in uploads:
            original = secure_filename(upload.filename or "")
            suffix = Path(original).suffix.casefold()
            if not original or suffix not in ALLOWED:
                raise AttachmentError("Định dạng tệp không được hỗ trợ.")
            data = upload.stream.read(MAX_FILE_BYTES + 1)
            if not data:
                raise AttachmentError(f"Tệp {original} rỗng.")
            if len(data) > MAX_FILE_BYTES:
                raise AttachmentError(f"Tệp {original} vượt giới hạn 8 MB.")
            mime = _verify_signature(suffix, data, original)
            text, extraction_status = _extract(suffix, data)
            safe_id = uuid.uuid4().hex
            storage_name = safe_id + suffix
            prepared.append({
                "id": safe_id,
                "conversation_id": conversation_id,
                "original_name": original[:180],
                "mime_type": mime,
                "size_bytes": len(data),
                "storage_name": storage_name,
                "extracted_text": text[:MAX_EXTRACTED_CHARS],
                "extraction_status": extraction_status,
                "_data": data,
            })
        saved = []
        created_paths = []
        directory = self.root / conversation_id
        directory.mkdir(parents=True, exist_ok=True)
        try:
            for item in prepared:
                storage_name = item["storage_name"]
                data = item.pop("_data")
                target = (directory / storage_name).resolve()
                if directory.resolve() not in target.parents:
                    raise AttachmentError("Đường dẫn lưu tệp không hợp lệ.")
                created_paths.append(target)
                target.write_bytes(data)
                saved.append(item)
        except Exception:
            for target in created_paths:
                target.unlink(missing_ok=True)
            raise
        return saved

    def discard(self, attachments):
        for item in attachments:
            target = self.path_for(item["conversation_id"], item["storage_name"])
            if target:
                target.unlink(missing_ok=True)

    def path_for(self, conversation_id, storage_name):
        directory = (self.root / conversation_id).resolve()
        target = (directory / storage_name).resolve()
        if directory not in target.parents or not target.is_file():
            return None
        return target

    def delete_conversation(self, conversation_id):
        directory = (self.root / conversation_id).resolve()
        if self.root not in directory.parents:
            return
        if directory.exists():
            for path in directory.iterdir():
                if path.is_file():
                    path.unlink(missing_ok=True)
            directory.rmdir()


def _verify_signature(suffix, data, filename):
    signatures = {
        ".png": data.startswith(b"\x89PNG\r\n\x1a\n"),
        ".jpg": data.startswith(b"\xff\xd8\xff"),
        ".jpeg": data.startswith(b"\xff\xd8\xff"),
        ".webp": len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP",
        ".pdf": data.startswith(b"%PDF-"),
        ".docx": _is_docx(data),
    }
    if suffix in signatures and not signatures[suffix]:
        raise AttachmentError(f"Nội dung tệp {filename} không khớp với phần mở rộng.")
    if suffix in {".png", ".jpg", ".jpeg", ".webp"}:
        try:
            from PIL import Image
            with Image.open(io.BytesIO(data)) as image:
                if image.width * image.height > 20_000_000:
                    raise AttachmentError(f"Ảnh {filename} vượt giới hạn 20 megapixel.")
                image.verify()
        except ImportError:
            pass
        except AttachmentError:
            raise
        except Exception as error:
            raise AttachmentError(f"Ảnh {filename} bị hỏng hoặc không hợp lệ.") from error
    if suffix in {".txt", ".csv"}:
        try:
            text = data.decode("utf-8-sig")
        except UnicodeDecodeError as error:
            raise AttachmentError(f"Không đọc được mã hóa văn bản của {filename}; cần UTF-8.") from error
        if "\x00" in text:
            raise AttachmentError(f"Tệp {filename} không phải văn bản thuần.")
        return "text/csv" if suffix == ".csv" else "text/plain"
    return {
        ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".webp": "image/webp", ".pdf": "application/pdf",
        ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    }[suffix]


def _is_docx(data):
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            names = set(archive.namelist())
            total_uncompressed = sum(item.file_size for item in archive.infolist())
            return ("[Content_Types].xml" in names and "word/document.xml" in names
                    and total_uncompressed <= 16 * 1024 * 1024)
    except (zipfile.BadZipFile, OSError):
        return False


def _extract(suffix, data):
    if suffix in {".txt", ".csv"}:
        text = data.decode("utf-8-sig")
        return text[:MAX_EXTRACTED_CHARS], "extracted" if text.strip() else "empty"
    if suffix == ".docx":
        try:
            with zipfile.ZipFile(io.BytesIO(data)) as archive:
                document = archive.getinfo("word/document.xml")
                if document.file_size > 16 * 1024 * 1024:
                    raise AttachmentError("Tài liệu DOCX có nội dung giải nén quá lớn.")
                xml = ElementTree.fromstring(archive.read("word/document.xml"))
            text = "\n".join(node.text or "" for node in xml.iter() if node.tag.endswith("}t"))
            return text[:MAX_EXTRACTED_CHARS], "extracted" if text.strip() else "empty"
        except (zipfile.BadZipFile, KeyError, ElementTree.ParseError):
            raise AttachmentError("Không thể trích xuất văn bản từ DOCX.")
    if suffix == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError:
            return "", "pdf_reader_unavailable"
        try:
            reader = PdfReader(io.BytesIO(data), strict=False)
            text = "\n".join((page.extract_text() or "") for page in reader.pages[:30])
            return text[:MAX_EXTRACTED_CHARS], "extracted" if text.strip() else "pdf_no_text"
        except Exception:
            raise AttachmentError("Không thể đọc PDF; tệp có thể bị hỏng hoặc được bảo vệ.")
    if suffix in {".png", ".jpg", ".jpeg", ".webp"}:
        return _ocr(data)
    return "", "unsupported"


def select_relevant_text(text, question, max_chars=6000, chunk_chars=1200):
    """Select a few relevant text windows from a long user attachment."""
    terms = set(re.findall(r"[\wÀ-ỹ]{3,}", question.casefold()))
    units = []
    for paragraph in re.split(r"\n+", text or ""):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        if len(paragraph) <= chunk_chars:
            units.append(paragraph)
        else:
            units.extend(paragraph[index:index + chunk_chars] for index in range(0, len(paragraph), chunk_chars))
    if not units:
        return ""
    ranked = sorted(enumerate(units), key=lambda pair: (
        -len(terms & set(re.findall(r"[\wÀ-ỹ]{3,}", pair[1].casefold()))), pair[0]
    ))
    selected = []
    used = 0
    for _, unit in ranked:
        if used + len(unit) + 2 > max_chars:
            continue
        selected.append(unit)
        used += len(unit) + 2
    if not selected:
        return units[0][:max_chars]
    original_order = {unit: index for index, unit in enumerate(units)}
    selected.sort(key=lambda unit: original_order[unit])
    return "\n\n".join(selected)


def _ocr(data):
    try:
        import pytesseract
        from PIL import Image
        pytesseract.get_tesseract_version()
    except Exception:
        return "", "ocr_unavailable"
    try:
        languages = set(pytesseract.get_languages(config=""))
        language = "vie+eng" if {"vie", "eng"}.issubset(languages) else "eng" if "eng" in languages else None
        if not language:
            return "", "ocr_languages_unavailable"
        image = Image.open(io.BytesIO(data)).convert("RGB")
        text = re.sub(r"\s+", " ", pytesseract.image_to_string(image, lang=language))[:MAX_EXTRACTED_CHARS].strip()
        return text, "ocr_extracted" if text else "ocr_no_text"
    except Exception:
        return "", "ocr_failed"
