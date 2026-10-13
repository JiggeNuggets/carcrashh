"""Settings for the local vehicle crash monitoring prototype."""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# Set DJANGO_SECRET_KEY in any environment beyond local development.
SECRET_KEY = os.environ.get(
    "DJANGO_SECRET_KEY",
    "django-insecure-local-development-key-change-before-deployment",
)
DEBUG = os.environ.get("DJANGO_DEBUG", "1") == "1"
ALLOWED_HOSTS = [host.strip() for host in os.environ.get(
    "DJANGO_ALLOWED_HOSTS", "127.0.0.1,localhost"
).split(",") if host.strip()]

INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "cameras.apps.CamerasConfig",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

ROOT_URLCONF = "crash_monitor.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
]

WSGI_APPLICATION = "crash_monitor.wsgi.application"
ASGI_APPLICATION = "crash_monitor.asgi.application"

DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": BASE_DIR / "db.sqlite3",
    }
}

AUTH_PASSWORD_VALIDATORS = [
    {"NAME": "django.contrib.auth.password_validation.UserAttributeSimilarityValidator"},
    {"NAME": "django.contrib.auth.password_validation.MinimumLengthValidator"},
    {"NAME": "django.contrib.auth.password_validation.CommonPasswordValidator"},
    {"NAME": "django.contrib.auth.password_validation.NumericPasswordValidator"},
]

LANGUAGE_CODE = "en-us"
TIME_ZONE = "Asia/Manila"
USE_I18N = True
USE_TZ = True

STATIC_URL = "static/"
STATIC_ROOT = BASE_DIR / "staticfiles"

MEDIA_URL = "media/"
MEDIA_ROOT = BASE_DIR / "media"

# Retention limits keep this local prototype's recording storage bounded.
CAMERA_MAX_CLIPS_PER_CAMERA = int(os.environ.get("CAMERA_MAX_CLIPS_PER_CAMERA", "50"))
CAMERA_MAX_SNAPSHOTS_PER_CAMERA = int(os.environ.get("CAMERA_MAX_SNAPSHOTS_PER_CAMERA", "100"))

# Optional Milestone 3 AI runtime. Model downloads are disabled by default so a
# web worker never makes an unexpected network request during startup.
CAMERA_AI_MODEL_PATH = Path(
    os.environ.get("CAMERA_AI_MODEL_PATH", str(BASE_DIR / "yolo11n.pt"))
)
CAMERA_AI_ALLOW_MODEL_DOWNLOAD = os.environ.get("CAMERA_AI_ALLOW_MODEL_DOWNLOAD", "0") == "1"
CAMERA_AI_DEVICE = os.environ.get("CAMERA_AI_DEVICE", "auto")
CAMERA_AI_TEMPORAL_MODEL_PATH = os.environ.get("CAMERA_AI_TEMPORAL_MODEL_PATH", "")
CAMERA_AI_TEMPORAL_THRESHOLD = float(os.environ.get("CAMERA_AI_TEMPORAL_THRESHOLD", "0.75"))
CAMERA_AI_TEST_VIDEO_ROOT = Path(
    os.environ.get("CAMERA_AI_TEST_VIDEO_ROOT", str(BASE_DIR / "media" / "ai_test_videos"))
)

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

LOGIN_URL = "login"
LOGIN_REDIRECT_URL = "cameras:list"
LOGOUT_REDIRECT_URL = "login"

# Keep request logs free of camera URLs, which can contain credentials.
LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "handlers": {"console": {"class": "logging.StreamHandler"}},
    "loggers": {
        "cameras": {"handlers": ["console"], "level": "INFO", "propagate": False},
    },
}
