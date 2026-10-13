from django.db import migrations, models


class Migration(migrations.Migration):
    initial = True
    dependencies = []

    operations = [
        migrations.CreateModel(
            name="Camera",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("name", models.CharField(max_length=120)),
                ("source_type", models.CharField(choices=[("usb", "USB camera"), ("rtsp", "RTSP camera")], max_length=8)),
                ("usb_device_index", models.PositiveSmallIntegerField(blank=True, null=True)),
                ("rtsp_url", models.CharField(blank=True, max_length=2048)),
                ("location", models.CharField(blank=True, max_length=255)),
                ("enabled", models.BooleanField(default=True)),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                ("last_connection_status", models.CharField(choices=[("unknown", "Not tested"), ("connected", "Connected"), ("unavailable", "Unavailable"), ("error", "Error")], default="unknown", max_length=16)),
                ("last_connection_message", models.CharField(blank=True, max_length=255)),
                ("last_tested_at", models.DateTimeField(blank=True, null=True)),
            ],
            options={"ordering": ("name",)},
        ),
    ]
