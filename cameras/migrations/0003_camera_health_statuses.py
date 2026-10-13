from django.db import migrations, models


def normalize_legacy_statuses(apps, schema_editor):
    Camera = apps.get_model("cameras", "Camera")
    Camera.objects.filter(last_connection_status__in=("unknown", "unavailable")).update(
        last_connection_status="disconnected"
    )


class Migration(migrations.Migration):
    dependencies = [("cameras", "0002_camera_buffer_enabled_camera_buffer_seconds_and_more")]

    operations = [
        migrations.RunPython(normalize_legacy_statuses, migrations.RunPython.noop),
        migrations.AlterField(
            model_name="camera",
            name="buffer_seconds",
            field=models.PositiveSmallIntegerField(
                choices=[(15, "15 seconds"), (20, "20 seconds"), (30, "30 seconds")],
                default=15,
                help_text="Rolling buffer window in seconds",
            ),
        ),
        migrations.AlterField(
            model_name="camera",
            name="last_connection_status",
            field=models.CharField(
                choices=[
                    ("connecting", "Connecting"),
                    ("connected", "Connected"),
                    ("disconnected", "Disconnected"),
                    ("reconnecting", "Reconnecting"),
                    ("disabled", "Disabled"),
                    ("error", "Error"),
                ],
                default="disconnected",
                max_length=16,
            ),
        ),
    ]
