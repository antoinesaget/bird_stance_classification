#!/usr/bin/env bash
set -euo pipefail

CONTAINER="${LABEL_STUDIO_CONTAINER:-birds-label-studio}"

ADMIN_EMAIL="${LABEL_STUDIO_ADMIN_EMAIL:-admin@local}"
ADMIN_PASSWORD="${LABEL_STUDIO_ADMIN_PASSWORD:-}"

ADRIEN_EMAIL="${LABEL_STUDIO_ADRIEN_EMAIL:-adrien@local}"
ADRIEN_USERNAME="${LABEL_STUDIO_ADRIEN_USERNAME:-adrien}"
ADRIEN_PASSWORD="${LABEL_STUDIO_ADRIEN_PASSWORD:-}"

if [[ -z "${ADMIN_PASSWORD}" ]]; then
  echo "ERROR: LABEL_STUDIO_ADMIN_PASSWORD is required."
  exit 1
fi

if [[ -z "${ADRIEN_PASSWORD}" ]]; then
  echo "ERROR: LABEL_STUDIO_ADRIEN_PASSWORD is required."
  exit 1
fi

if ! docker ps --format '{{.Names}}' | grep -Fxq "${CONTAINER}"; then
  echo "ERROR: container '${CONTAINER}' is not running."
  exit 1
fi

docker exec \
  -e ADMIN_EMAIL="${ADMIN_EMAIL}" \
  -e ADMIN_PASSWORD="${ADMIN_PASSWORD}" \
  -e ADRIEN_EMAIL="${ADRIEN_EMAIL}" \
  -e ADRIEN_USERNAME="${ADRIEN_USERNAME}" \
  -e ADRIEN_PASSWORD="${ADRIEN_PASSWORD}" \
  "${CONTAINER}" \
  python /label-studio/label_studio/manage.py shell -c "
import os
from users.models import User

admin_email = os.environ['ADMIN_EMAIL']
admin_password = os.environ['ADMIN_PASSWORD']
adrien_email = os.environ['ADRIEN_EMAIL']
adrien_username = os.environ['ADRIEN_USERNAME']
adrien_password = os.environ['ADRIEN_PASSWORD']

admin = User.objects.filter(email=admin_email).first()
if admin is None:
    admin = User.objects.create_superuser(email=admin_email, password=admin_password)
else:
    admin.set_password(admin_password)
    admin.is_superuser = True
    admin.is_staff = True
    admin.is_active = True
    admin.save(update_fields=['password', 'is_superuser', 'is_staff', 'is_active'])

adrien = User.objects.filter(email=adrien_email).first()
if adrien is None:
    adrien = User.objects.create_user(email=adrien_email, password=adrien_password, username=adrien_username)
else:
    adrien.username = adrien_username
    adrien.set_password(adrien_password)
    adrien.is_active = True
    adrien.save(update_fields=['username', 'password', 'is_active'])

print(f'Admin ready: {admin.email}')
print(f'Adrien ready: username={adrien.username} email={adrien.email}')
"
