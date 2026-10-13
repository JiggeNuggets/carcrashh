from django.core.management.base import BaseCommand, CommandError

from cameras.ai_detection import DetectorUnavailable
from cameras.models import Camera
from cameras.video_testing import RecordedVideoProcessor


class Command(BaseCommand):
    help = "Run the AI pipeline on a video in the configured safe test-video directory."

    def add_arguments(self, parser):
        parser.add_argument("video", help="Filename or safe absolute path under CAMERA_AI_TEST_VIDEO_ROOT")
        parser.add_argument("--camera", type=int, required=True, help="Camera configuration ID")
        parser.add_argument("--display", action="store_true", help="Show an annotated OpenCV preview; press q to stop")

    def handle(self, *args, **options):
        try:
            camera = Camera.objects.get(pk=options["camera"])
        except Camera.DoesNotExist as exc:
            raise CommandError("Camera configuration not found.") from exc
        processor = RecordedVideoProcessor(camera)

        def report(frame_index, detections, tracks, score):
            if frame_index % max(1, camera.ai_frame_rate * 10) == 0:
                self.stdout.write(
                    f"frame={frame_index} objects={len(detections)} tracks={len(tracks)} suspicion={score:.2f}"
                )

        try:
            result = processor.process(
                options["video"],
                display=options["display"],
                progress_callback=report,
            )
        except (ValueError, DetectorUnavailable) as exc:
            raise CommandError(str(exc)) from exc
        self.stdout.write(self.style.SUCCESS(
            "Processed {frames_read} frames ({inference_frames} inference frames) in "
            "{elapsed_seconds}s; created {incidents_created} suspected incident(s).".format(**result)
        ))
