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
from django.db.models import CharField, F, Value
from django.http import HttpRequest, HttpResponse, HttpResponseForbidden, HttpResponseNotFound, HttpResponseNotModified
from django.utils._os import safe_join
from django.utils.http import http_date
from drf_spectacular.utils import extend_schema
from io_storages.localfiles.models import LocalFilesImportStorage
from ranged_fileresponse import RangedFileResponse
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated

logger = logging.getLogger(__name__)


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
        return not_modified

    content_type, _ = mimetypes.guess_type(str(full_path))
    content_type = content_type or 'application/octet-stream'

    response = RangedFileResponse(request, file_handle, content_type=content_type)
    response['ETag'] = etag
    response['Last-Modified'] = last_modified
    response['Cache-Control'] = cache_control
    return response


@extend_schema(exclude=True)
@api_view(['GET'])
@permission_classes([IsAuthenticated])
def localfiles_data(request):
    """Serve files residing under LocalFilesImportStorage roots with ETag support."""
    path = request.GET.get('d')
    if settings.LOCAL_FILES_SERVING_ENABLED is False:
        return HttpResponseForbidden(
            "Serving local files can be dangerous, so it's disabled by default. "
            'You can enable it with LOCAL_FILES_SERVING_ENABLED environment variable, '
            'please check docs: https://labelstud.io/guide/storage.html#Local-storage'
        )

    local_serving_document_root = settings.LOCAL_FILES_DOCUMENT_ROOT
    if path and request.user.is_authenticated:
        path = posixpath.normpath(path).lstrip('/')
        full_path = Path(safe_join(local_serving_document_root, path))
        user_has_permissions = False

        full_path_dir = os.path.normpath(os.path.dirname(str(full_path)))
        localfiles_storage = LocalFilesImportStorage.objects.annotate(
            _full_path=Value(full_path_dir, output_field=CharField())
        ).filter(_full_path__startswith=F('path'))
        if localfiles_storage.exists():
            user_has_permissions = any(storage.project.has_permission(request.user) for storage in localfiles_storage)

        if user_has_permissions and os.path.exists(full_path):
            return build_localfile_response(
                request=request,
                full_path=str(full_path),
                if_none_match_header=request.META.get('HTTP_IF_NONE_MATCH'),
            )
        else:
            return HttpResponseNotFound()

    return HttpResponseForbidden()
