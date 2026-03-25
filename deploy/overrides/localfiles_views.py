"""
HTTP views dedicated to LocalFiles storage download operations.

Override note:
- This file mirrors Label Studio's localfiles view with one UX-oriented change:
  add explicit cache headers for browser-side thumbnail reuse in grid view.
"""
import logging
import mimetypes
import os
import posixpath
from pathlib import Path
from typing import Optional

from django.conf import settings
from django.http import HttpRequest, HttpResponse, HttpResponseForbidden, HttpResponseNotFound, HttpResponseNotModified
from django.utils._os import safe_join
from django.utils.http import http_date
from drf_spectacular.utils import extend_schema
from io_storages.localfiles.models import LocalFilesImportStorage
from ranged_fileresponse import RangedFileResponse
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated

logger = logging.getLogger(__name__)


def _append_vary(response: HttpResponse, value: str) -> None:
    current = response.get('Vary')
    if not current:
        response['Vary'] = value
        return
    parts = [part.strip() for part in current.split(',') if part.strip()]
    if value not in parts:
        response['Vary'] = f'{current}, {value}'


def _apply_cors_headers(response: HttpResponse, request: HttpRequest) -> HttpResponse:
    origin = request.headers.get('Origin') or request.META.get('HTTP_ORIGIN')
    response['Access-Control-Allow-Origin'] = origin if origin else '*'
    response['Access-Control-Allow-Credentials'] = 'true'
    response['Access-Control-Allow-Methods'] = 'GET, OPTIONS'
    response['Access-Control-Allow-Headers'] = 'Authorization, Content-Type, Range'
    response['Access-Control-Expose-Headers'] = 'Content-Length, Content-Range, Accept-Ranges, ETag, Last-Modified'
    if origin:
        _append_vary(response, 'Origin')
    return response


def _is_same_or_subpath(path: str, root: str) -> bool:
    path_norm = os.path.normcase(os.path.realpath(os.path.normpath(path)))
    root_norm = os.path.normcase(os.path.realpath(os.path.normpath(root)))
    return path_norm == root_norm or path_norm.startswith(root_norm + os.sep)


def _user_has_localfiles_access(request: HttpRequest, full_path: Path, local_serving_document_root: str) -> bool:
    full_path_str = str(full_path)
    full_path_dir = os.path.dirname(full_path_str)

    storages = list(LocalFilesImportStorage.objects.select_related('project').all())
    if storages:
        for storage in storages:
            storage_path = (storage.path or '').strip()
            if not storage_path:
                continue
            if _is_same_or_subpath(full_path_dir, storage_path) and storage.project.has_permission(request.user):
                return True
        return False

    # Fallback for restored environments where local-files storage rows are absent.
    default_allowed_root = os.path.join(local_serving_document_root, 'birds_project')
    return _is_same_or_subpath(full_path_str, default_allowed_root)


def _if_none_match_satisfied(header_value: Optional[str], etag: str) -> bool:
    """Return True if the client's cached representation matches the current file."""
    if not header_value:
        return False
    etag_candidates = [candidate.strip() for candidate in header_value.split(',') if candidate.strip()]
    return '*' in etag_candidates or etag in etag_candidates


def build_localfile_response(
    request: HttpRequest,
    full_path: str,
    if_none_match_header: Optional[str],
) -> HttpResponse:
    """
    Stream the requested file and attach cache validators.

    We keep weak ETag semantics from upstream and also provide explicit cache
    policy headers so remounted grid thumbnails can reuse browser cache.
    """
    try:
        file_handle = open(full_path, mode='rb')
    except OSError as exc:
        logger.error('Error opening file %s: %s', full_path, exc)
        return HttpResponseNotFound(f'Error opening file {full_path}')

    stat_result = os.fstat(file_handle.fileno())
    mtime_ns = getattr(stat_result, 'st_mtime_ns', int(stat_result.st_mtime * 1_000_000_000))
    etag = f'W/"{mtime_ns:x}-{stat_result.st_size:x}"'
    last_modified = http_date(stat_result.st_mtime)
    cache_control = 'private, max-age=3600, must-revalidate'

    if _if_none_match_satisfied(if_none_match_header, etag):
        file_handle.close()
        not_modified = HttpResponseNotModified()
        not_modified['ETag'] = etag
        not_modified['Last-Modified'] = last_modified
        not_modified['Cache-Control'] = cache_control
        return _apply_cors_headers(not_modified, request)

    content_type, _ = mimetypes.guess_type(str(full_path))
    content_type = content_type or 'application/octet-stream'

    response = RangedFileResponse(request, file_handle, content_type=content_type)
    response['ETag'] = etag
    response['Last-Modified'] = last_modified
    response['Cache-Control'] = cache_control
    return _apply_cors_headers(response, request)


@extend_schema(exclude=True)
@api_view(['GET', 'OPTIONS'])
@permission_classes([IsAuthenticated])
def localfiles_data(request):
    """Serve files residing under LocalFilesImportStorage roots with ETag support."""
    if request.method == 'OPTIONS':
        return _apply_cors_headers(HttpResponse(status=204), request)

    path = request.GET.get('d')
    if settings.LOCAL_FILES_SERVING_ENABLED is False:
        return _apply_cors_headers(HttpResponseForbidden(
            "Serving local files can be dangerous, so it's disabled by default. "
            'You can enable it with LOCAL_FILES_SERVING_ENABLED environment variable, '
            'please check docs: https://labelstud.io/guide/storage.html#Local-storage'
        ), request)

    local_serving_document_root = settings.LOCAL_FILES_DOCUMENT_ROOT
    if path and request.user.is_authenticated:
        path = posixpath.normpath(path).lstrip('/')
        full_path = Path(safe_join(local_serving_document_root, path))
        user_has_permissions = _user_has_localfiles_access(request, full_path, local_serving_document_root)
        if user_has_permissions and os.path.exists(full_path):
            return build_localfile_response(
                request=request,
                full_path=str(full_path),
                if_none_match_header=request.META.get('HTTP_IF_NONE_MATCH'),
            )
        return _apply_cors_headers(HttpResponseNotFound(), request)

    return _apply_cors_headers(HttpResponseForbidden(), request)
