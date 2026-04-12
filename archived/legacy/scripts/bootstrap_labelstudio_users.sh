#!/usr/bin/env bash
set -euo pipefail

CONTAINER="${LABEL_STUDIO_CONTAINER:-birds-label-studio}"

ADMIN_EMAIL="${LABEL_STUDIO_ADMIN_EMAIL:-admin@local}"
ADMIN_PASSWORD="${LABEL_STUDIO_ADMIN_PASSWORD:-}"

ADRIEN_EMAIL="${LABEL_STUDIO_ADRIEN_EMAIL:-adrien@local}"
ADRIEN_USERNAME="${LABEL_STUDIO_ADRIEN_USERNAME:-adrien}"
ADRIEN_PASSWORD="${LABEL_STUDIO_ADRIEN_PASSWORD:-}"
ADRIEN_LOGIN="${LABEL_STUDIO_ADRIEN_LOGIN:-${ADRIEN_EMAIL}}"

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
  -e ADRIEN_LOGIN="${ADRIEN_LOGIN}" \
  "${CONTAINER}" \
  python /label-studio/label_studio/manage.py shell -c "
import os
from django.contrib.auth import authenticate
from users.models import User

admin_email = os.environ['ADMIN_EMAIL']
admin_password = os.environ['ADMIN_PASSWORD']
adrien_email = os.environ['ADRIEN_EMAIL']
adrien_username = os.environ['ADRIEN_USERNAME']
adrien_password = os.environ['ADRIEN_PASSWORD']
adrien_login = os.environ['ADRIEN_LOGIN']
username_field = getattr(User, 'USERNAME_FIELD', 'username')

if username_field == 'email' and '@' not in adrien_login:
    print('WARNING: USERNAME_FIELD=email, but ADRIEN_LOGIN is not an email; falling back to ADRIEN_EMAIL')
    adrien_login = adrien_email

admin = User.objects.filter(email=admin_email).first()
if admin is None:
    admin = User.objects.create_superuser(email=admin_email, password=admin_password)
else:
    admin.set_password(admin_password)
    admin.is_superuser = True
    admin.is_staff = True
    admin.is_active = True
    admin.save(update_fields=['password', 'is_superuser', 'is_staff', 'is_active'])

if username_field == 'email':
    adrien = User.objects.filter(email=adrien_login).first()
    if adrien is None and adrien_email != adrien_login:
        adrien = User.objects.filter(email=adrien_email).first()
    if adrien is None:
        adrien = User.objects.filter(username=adrien_username).first()
else:
    adrien = User.objects.filter(username=adrien_username).first()
    if adrien is None:
        adrien = User.objects.filter(email=adrien_email).first()
if adrien is None:
    new_user_email = adrien_login if username_field == 'email' else adrien_email
    adrien = User.objects.create_user(
        email=new_user_email,
        password=adrien_password,
        username=adrien_username,
        is_active=True,
    )

adrien.email = adrien_login if username_field == 'email' else adrien_email
adrien.username = adrien_username
adrien.set_password(adrien_password)
adrien.is_active = True
update_fields = ['email', 'username', 'password', 'is_active']
# Label Studio signup flow may set additional flags; mirror them if present.
for field_name in ('invite_activated', 'social_auth_finished', 'email_verified'):
    if hasattr(adrien, field_name):
        setattr(adrien, field_name, True)
        update_fields.append(field_name)
adrien.save(update_fields=update_fields)

auth_identifier = adrien_login if username_field == 'email' else adrien_username
auth_check = authenticate(username=auth_identifier, password=adrien_password) is not None

print(f'Admin ready: {admin.email}')
print(f'Adrien ready: username={adrien.username} email={adrien.email}')
print(f'Auth mode: USERNAME_FIELD={username_field} login_identifier={auth_identifier}')
print(f'Adrien auth check: {auth_check}')
if not auth_check:
    raise SystemExit(f'ERROR: Adrien login check failed for identifier={auth_identifier}')
"
