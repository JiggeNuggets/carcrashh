# CrashGuard next-step research and implementation report

**Prepared:** 19 October 2026  
**Scope:** Car-focused crash detection, dispatcher-to-responder workflow, alerts/audio, reports, and administrator analytics  
**Current system:** Django 6 application with camera management, live USB/RTSP capture, YOLO vehicle detection, ByteTrack-style tracking, heuristic crash scoring, evidence snapshots/clips, incident pages, and an optional GRU/ONNX temporal-classifier pipeline.

## 1. Executive recommendation

Do not connect raw AI detections directly to emergency responders. The next release should be a **human-in-the-loop incident operations system**:

1. AI creates a *suspected incident* and preserves evidence.
2. A dispatcher receives a visible and audible alert.
3. The dispatcher reviews the clip, confirms or dismisses the event, assigns priority, and records a location.
4. Only a dispatcher-confirmed incident can create a dispatch.
5. Selected responders are notified and must acknowledge or decline.
6. The dispatcher tracks response progress through `assigned`, `acknowledged`, `en_route`, `on_scene`, and `resolved`.
7. Every transition, notification attempt, acknowledgement, edit, and evidence access is audited.
8. Reports and analytics use confirmed operational data, while model evaluation separately counts false alarms and missed incidents.

This order reduces the largest current risk: treating an uncalibrated heuristic score as a confirmed crash. Real public-safety or Emergency 911 integration should remain disabled until an authorized agency approves the workflow, security review, operating procedures, and test environment.

## 2. What already exists and what is missing

### Existing and reusable

- `Camera`, `RecordedClip`, `CameraSnapshot`, and `SuspectedIncident` database models.
- Protected login, camera configuration, recordings, incident list, and incident detail pages.
- YOLO detection for car, motorcycle, bus, and truck classes.
- Track histories, velocity/direction features, heuristic collision evidence, cooldown, and evidence generation.
- A small GRU design using 16 normalized motion features over 12 sampled frames.
- Scripts for manifests, feature extraction, training, ONNX export, and a synthetic functional smoke test.
- Dataset metadata already audited locally for CCD, DoTA, and CADP.

### Missing before an operational pilot

- Dispatcher and responder roles and permissions.
- A dispatcher review form and immutable review history.
- Assignment, acknowledgement, escalation, and resolution workflow.
- Durable background jobs and notification delivery records.
- Real-time incident queue updates.
- Operator-controlled audio and desktop alerts.
- Structured incident reports and exports.
- Audit trail, model-version traceability, retention policy, and privacy controls.
- Admin analytics for operations, cameras, alerts, and model quality.
- Fixed-camera validation data from the intended deployment sites.

## 3. Target workflow

```text
camera -> vehicle detection/tracking -> temporal risk logic
       -> suspected incident + snapshot + pre/post clip
       -> dispatcher queue + visual/audio alert
       -> human review
          |-- dismiss -> false-positive reason -> model evaluation set
          `-- confirm -> set priority/location -> create dispatch
                         -> notify selected responders
                         -> acknowledge/decline
                         -> en route -> on scene -> resolved
                         -> final report + audit history + analytics
```

Use two separate state machines so evidence processing, human review, and field response cannot contradict one another.

| Concern | Recommended states |
|---|---|
| Evidence processing | `detected`, `recording`, `ready`, `media_error` (already present) |
| Human review | `pending`, `in_review`, `confirmed`, `dismissed`, `needs_more_information` |
| Response | `not_dispatched`, `dispatching`, `assigned`, `acknowledged`, `en_route`, `on_scene`, `resolved`, `cancelled` |

The AI `suspicion_score` must remain distinct from dispatcher priority and confirmed severity. A score is model evidence; it is not injury, damage, or emergency severity.

## 4. Dataset research and acquisition plan

### Recommended datasets, in priority order

| Priority | Dataset and official link | Viewpoint / contents | Best use in CrashGuard | Important limitation |
|---:|---|---|---|---|
| 1 | [AI City Challenge 2021, Track 4](https://www.aicitychallenge.org/2021-data-and-evaluation/) and [dataset access](https://www.aicitychallenge.org/ai-city-challenge-dataset-access/) | Fixed traffic cameras; 100 training and 150 test videos of about 15 minutes; crashes and stalled vehicles | Primary external benchmark for the current CCTV/RTSP deployment style; long videos also measure false alarms per camera-hour | Labels combine crashes and stalled vehicles; access and challenge terms must be reviewed |
| 2 | [CADP](https://ankitshah009.github.io/accident_forecasting_traffic_camera.html) | Fixed CCTV; authors describe 1,416 segments, with a spatially/temporally annotated subset | Fixed-camera localization and temporal onset evaluation | Non-commercial research conditions and permission requirements apply |
| 3 | [TUMTraf-A / Accid3nD](https://tum-traffic-dataset.github.io/tumtraf-a/) | Real roadside highway accident data with richer roadside perception annotations | Strong later-stage out-of-domain validation and, where applicable, geometry-aware research | Dataset terms are restrictive; the TUMTraf dev-kit states CC BY-NC-ND 4.0 for dataset releases—verify the exact TUMTraf-A release terms |
| 4 | [Car Crash Dataset (CCD)](https://github.com/Cogito2012/CarCrashDataset) | 1,500 crash and 3,000 normal 5-second dashcam clips at 10 FPS, with temporal crash labels and conditions | Start the existing GRU training pipeline and test day/night/weather robustness | Dashcam-to-CCTV domain mismatch; repository license does not automatically settle rights to source footage |
| 5 | [DoTA](https://github.com/MoonBlvd/Detection-of-Traffic-Anomaly) | 4,677 dashcam anomaly clips with temporal class/object annotations and supplied tracks/features | Broader anomaly pretraining and hard-negative mining | About 55 GB; primarily dashcam; verify footage terms separately from the MIT code license |
| 6 | [Nexar Collision Prediction](https://huggingface.co/datasets/nexar-ai/nexar_collision_prediction) | 31.4 GB dashcam collision-prediction dataset with event/alert timing and scene/weather metadata | Modern anticipation benchmark and condition-based evaluation | Nexar Open Data License; dashcam viewpoint differs from deployment cameras |
| 7 | [Dashcam Accident Dataset (DAD)](https://aliensunmin.github.io/project/dashcam/) | 620 videos from six Taiwanese cities; car-car, motorcycle-car, and other collisions | Secondary anticipation benchmark and regional traffic diversity | Older, small, dashcam-specific dataset; obtain and retain its research agreement/terms |
| 8 | [BDD100K](https://bdd-data.berkeley.edu/) and [official toolkit/license](https://github.com/bdd100k/bdd100k/blob/master/doc/source/license.rst) | Large diverse driving dataset with cars and normal driving | Vehicle-detector fine-tuning only if needed; normal/hard-negative examples | Not a crash dataset; data use is limited by its separate license, especially for commercial use |

### Dataset decision

- **For an academic prototype now:** run the existing CCD pipeline, then add DoTA.
- **For this product architecture:** make AI City Track 4 and CADP the main external evaluation sources because their fixed-camera viewpoint resembles USB/RTSP traffic cameras.
- **For a credible pilot:** collect a lawful, consented or otherwise lawfully processed local dataset from the exact cameras. Public datasets cannot replace local validation across camera angle, compression, night lighting, rain, congestion, jeepneys, motorcycles, and occlusion.
- **For car focus:** retain detector classes `car`, `truck`, and `bus`; treat motorcycles as a separate safety subgroup rather than discarding them. Report metrics separately for car-car, car-motorcycle, single-vehicle, stalled vehicle, and non-crash congestion.

### Required dataset registry

Create `DatasetSource` and `DatasetAsset` records or a version-controlled registry containing:

- source URL, owner, version/download date, checksum, and storage location;
- license/terms file and approved use (research, internal evaluation, commercial training);
- viewpoint (`fixed_cctv`, `dashcam`, `synthetic`), geography, weather, and lighting;
- annotation schema and known quality issues;
- original-video or camera group ID for leakage-safe splitting;
- whether redistribution and derived-model use are allowed.

Do not download or combine footage until the project owner accepts each dataset's terms. Code licenses and video rights are not the same thing.

## 5. Model training and evaluation process

### Step 1 — Freeze an honest baseline

1. Save the current heuristic configuration and source commit.
2. Run the current unit suite and recorded-video command on an approved validation sample.
3. Store per-event predictions, camera, timestamp, threshold, model version, and reviewer outcome.
4. Establish baseline metrics before replacing any component.

### Step 2 — Define labels before training

Use an annotation guide with:

- `normal`, `near_miss`, `collision`, `stalled_vehicle`, and `uncertain` event labels;
- first visible anomaly time, collision/impact time, and event end time;
- vehicles involved and vehicle categories;
- occlusion, camera shake, rain, night, congestion, and stopped-traffic tags;
- reviewer ID, review timestamp, and disagreement/adjudication fields.

Two reviewers should independently label a validation subset. Resolve disagreements before using it as ground truth.

### Step 3 — Build leakage-safe splits

- Split by original source video for internet datasets.
- Split local data by **camera and date block**, not random frames.
- Keep the final test set locked until model and threshold decisions are complete.
- Include long normal videos. Short, pre-trimmed crash clips create unrealistic class priors and hide false alarms.

### Step 4 — Keep detection and crash recognition separate

The current architecture is correct in principle:

- YOLO detects road users.
- Tracking turns detections into motion histories.
- The temporal model estimates event risk from sequences.
- Conservative rules combine trajectory evidence and model support.

Do not fine-tune YOLO using CCD's supplied detector outputs or DoTA anomaly regions as if they were human-labeled vehicle boxes. If vehicle detection is weak on local footage, label real boxes from local cameras or use a properly licensed vehicle-detection dataset such as BDD100K.

### Step 5 — Run the existing training pipeline

Preferred route: use `ml/CrashGuard_Milestone_3_1_Colab.ipynb`, which includes Drive storage, deterministic grouped splits, checkpoint resume, threshold selection, held-out metrics, and ONNX export.

Equivalent local stages after footage is lawfully acquired:

```powershell
# Validate the software path first; synthetic metrics are not model evidence.
.\.venv\Scripts\python.exe ml\training\smoke_test.py --output ml\smoke_output

# Build manifests. Repeat with --source dota for DoTA.
.\.venv\Scripts\python.exe ml\training\prepare_manifests.py `
  --source ccd `
  --root C:\approved-data\CarCrash `
  --output C:\approved-data\manifests\ccd.csv

# Cache production-compatible detector/tracker features for each manifest.
.\.venv\Scripts\python.exe ml\training\extract_features.py `
  --manifest C:\approved-data\manifests\ccd.csv `
  --output C:\approved-data\features\ccd `
  --model C:\carrcrash\yolo11n.pt `
  --sample-fps 2 `
  --image-size 320 `
  --confidence 0.45

# Train and export the temporal classifier. Add --resume to continue last.pt.
.\.venv\Scripts\python.exe ml\training\train_temporal.py `
  --features C:\approved-data\features\ccd `
  --output C:\approved-data\runs\temporal-v1 `
  --sequence-length 12 `
  --stride 2 `
  --epochs 30 `
  --batch-size 256

# Copy both crash_temporal.onnx and crash_temporal.json into a versioned model folder.
$env:CAMERA_AI_TEMPORAL_MODEL_PATH = "C:\carrcrash\models\v1\crash_temporal.onnx"
.\.venv\Scripts\python.exe manage.py check
```

### Step 6 — Evaluate for operations, not only accuracy

Required held-out metrics:

- event precision, recall, F1, and precision-recall AUC;
- false alarms per camera-hour and per normal video;
- missed crashes per subgroup;
- median/p95 detection delay and mean time-to-accident for anticipation;
- performance by camera, day/night, rain, congestion, and vehicle pairing;
- percentage of events with usable evidence;
- end-to-end processing latency and inference FPS.

Accuracy alone is unsuitable for rare crashes. Select thresholds on validation data using an explicit operational target such as “no more than X false alerts per camera-hour while maximizing recall,” then report the untouched test result.

### Step 7 — Shadow pilot before live dispatch

1. Run alerts silently for two to four weeks.
2. Dispatchers label candidates without notifying responders.
3. Measure workload, false alarms, missed incidents found by manual review, and evidence quality.
4. Calibrate thresholds per camera only when each camera has enough labeled data.
5. Promote a signed model version only after approval; retain rollback to the previous version.

## 6. Dispatcher and responder implementation

### Roles

Use Django Groups and model permissions:

| Role | Minimum permissions |
|---|---|
| Administrator | users/roles, cameras, model activation, retention, integrations, all analytics |
| Dispatcher | view evidence, claim/review incidents, confirm/dismiss, dispatch, cancel/escalate, complete reports |
| Responder | view only assigned incident details, acknowledge/decline, update response status, add notes |
| Auditor | read-only incidents, reports, delivery history, model versions, and audit events |

Django's official authentication documentation supports assigning permissions through groups: [Django authentication and groups](https://docs.djangoproject.com/en/6.0/topics/auth/default/).

### New data model

Implement these models in a new `operations` app so camera/ML concerns remain isolated:

| Model | Key fields |
|---|---|
| `IncidentReview` | incident, reviewer, decision, priority, confirmed severity, notes, dismissal reason, timestamps |
| `ResponderProfile` | user, unit/team, contact channels, availability, service area, active flag |
| `Dispatch` | incident, dispatcher, status, priority, location text/coordinates, instructions, timestamps, version |
| `DispatchAssignment` | dispatch, responder, status, assigned/acknowledged/declined/en-route/on-scene timestamps, decline reason |
| `NotificationDelivery` | assignment, channel, destination fingerprint, status, provider ID, attempt count, last error, sent/delivered/read timestamps, idempotency key |
| `IncidentReport` | incident, author, narrative, actions taken, outcome, injuries/damage as `unknown` by default, finalized/version timestamps |
| `AuditEvent` | actor, action, object type/ID, before/after JSON, IP/session metadata, timestamp |
| `AlertPreference` | user, sound enabled, volume, desktop notifications, quiet-mode policy |
| `ModelRelease` | name/version, file checksum, feature schema, training datasets, metrics, threshold, approved/activated by and at |

Add database constraints so there is at most one active review claim per incident, status transitions are validated, notification idempotency keys are unique, and finalized reports cannot be silently overwritten.

### Dispatcher queue UI

Build `/operations/dispatch/` with:

- a top “new/unclaimed” lane ordered by priority and age;
- camera, location, detected time, evidence readiness, score, and age-at-a-glance;
- claim button to prevent two dispatchers acting on one incident;
- evidence viewer with normal video/snapshot first and optional overlay separately;
- actions: `Confirm`, `Dismiss`, `Need more information`;
- required dismissal reason and optional model-feedback note;
- confirmed-incident form for priority, location, responder selection, and instructions;
- response timeline and delivery/acknowledgement state;
- conspicuous “AI-generated candidate—not a confirmed crash” text until confirmed.

Use short polling first (for example, every 3–5 seconds) because it is easy to test and operate. Add WebSockets only after the state transitions are stable.

### Responder UI

Build `/operations/respond/` as a mobile-friendly page:

- only the responder's active assignments;
- minimal necessary location, priority, dispatcher instructions, and safe evidence link;
- large `Acknowledge`, `Decline`, `En route`, `On scene`, and `Resolved` controls;
- required reason for decline/cancellation;
- server timestamps rather than client timestamps;
- no access to unrelated cameras or incidents.

### Dispatch logic

1. A confirmed review creates one `Dispatch` in a database transaction.
2. Lock the incident/dispatch row with `select_for_update()` during transitions.
3. Create assignments and an outbox/delivery row for every recipient.
4. Enqueue notification jobs only with `transaction.on_commit()`.
5. Use the unique idempotency key `dispatch_id + responder_id + channel + event_type`.
6. Retry temporary provider failures with bounded exponential backoff.
7. If no responder acknowledges within the configured SLA, alert the dispatcher and optionally notify the next authorized tier.
8. Stop escalation when one or the required number of responders acknowledges.
9. Never infer acknowledgement from “message sent”; acknowledgement is an explicit responder action.
10. Every transition writes an `AuditEvent`.

For production jobs, replace ad-hoc threads with a durable queue such as Celery plus a supported broker. Celery's Django integration recommends enqueueing after database commit, and its task system supports explicit retry behavior: [Celery with Django](https://docs.celeryq.dev/en/stable/django/first-steps-with-django.html).

### Notification channels

Implement in this order:

1. **In-app alert** — authoritative and fully testable locally.
2. **Email** — useful for non-urgent summaries and fallback.
3. **SMS or approved messaging provider** — only after credentials, cost controls, delivery callbacks, and privacy review.
4. **Agency/E911 integration** — only under a formal agreement and the agency's approved interface; never simulate a production emergency dispatch.

Messages should contain the incident ID, priority, broad location, action link, and expiry. Do not put sensitive evidence, RTSP credentials, or unnecessary personal data directly in SMS/email.

## 7. Audio and attention design

Treat “audio” initially as dispatcher/responder alert audio, not microphone-based crash classification.

### Alert audio MVP

- Add an explicit `Enable alert sound` control after login; persist the preference per user.
- Use three short, distinguishable patterns: new unreviewed event, escalation/SLA breach, and delivery/system failure.
- Pair every sound with color-independent text, icon, and desktop/in-app notification.
- Provide mute, volume, test-sound, and “acknowledge alert” controls.
- Do not loop continuously; repeat on a controlled escalation schedule.
- Log alert creation and acknowledgement, but not the user's local speaker behavior.

Browsers commonly block audible autoplay until the user interacts with the page, so the enable/test control is required. Desktop notifications also require permission and generally HTTPS outside local development: [MDN autoplay guidance](https://developer.mozilla.org/en-US/docs/Web/Media/Guides/Autoplay) and [MDN Notifications API](https://developer.mozilla.org/en-US/docs/Web/API/Notifications_API/Using_the_Notifications_API).

### Microphone/audio crash detection (later research only)

Do not add this to the MVP. It requires microphones, synchronized audio/video, lawful audio collection, a representative labeled acoustic dataset, noise/wind/horn suppression, and separate privacy review. Audio should be supporting evidence only, never the sole trigger for dispatch. Philippine privacy and wiretapping implications require legal review before recording public audio.

## 8. Reports UI

### Incident report screen

Create `/operations/incidents/<uuid>/report/` with:

- identifiers, camera, verified location, detection/confirmation/dispatch timestamps;
- AI evidence and model version/threshold used;
- dispatcher review and reason;
- responder assignments and complete status timeline;
- notification attempt/delivery/acknowledgement history;
- narrative, actions taken, outcome, and attachments;
- `unknown`, `none reported`, and `confirmed` choices for injuries/damage—never default either to true;
- author, reviewer, finalized timestamp, and amendments.

Generate printable HTML first. Add PDF export only after the fields stabilize. Exports must carry report ID, revision, generation time, and a confidentiality footer.

### Search and exports

Filters:

- date/time range, camera/location, review decision, priority, response status;
- responder/team, model version, severity, and dismissal reason;
- evidence available/error and notification delivery state.

Allow CSV export of aggregate/operational fields to authorized administrators or auditors. Evidence files should remain protected downloads and should not be embedded into bulk exports.

## 9. Administrator analytics

Build a custom `/analytics/` dashboard; keep Django admin for record maintenance rather than the operational dashboard.

### Operations cards

- suspected, confirmed, dismissed, dispatched, and resolved incidents;
- median and p95 detection-to-review time;
- median and p95 confirm-to-dispatch time;
- median and p95 dispatch-to-acknowledgement/on-scene time;
- open incidents by age and SLA breaches;
- notification success/failure by channel.

### Model-quality cards

- dispatcher-confirmed precision proxy: `confirmed / reviewed candidates`;
- dismissal rate and dismissal reasons by camera;
- false alerts per camera-hour;
- confirmed events and misses from audited footage;
- score distribution and threshold by camera/model version;
- subgroup breakdown: lighting, weather, congestion, and vehicle pairing.

Dispatcher confirmation alone cannot measure recall because the UI only sees generated candidates. Estimate recall through periodic manual audits of randomly sampled footage and known-event test sets.

### Camera/system-health cards

- connected/disconnected cameras and uptime;
- inference FPS, queue drops, reconnects, and media-generation errors;
- storage use, evidence retention/deletion counts, and worker/job failures;
- model version coverage and cameras running fallback heuristics.

Start with server-rendered tables and small charts. Pre-aggregate daily metrics in a scheduled job after query volume makes live aggregation slow.

## 10. Privacy, security, and governance

For a Philippine deployment, CCTV footage is personal-data processing. The [National Privacy Commission's Circular No. 2024-02](https://privacy.gov.ph/wp-content/uploads/2024/08/NPC-Circular-No.-2024-02-CCTV-Systems.pdf) covers CCTV systems and calls for transparency, appropriate security, operating policies, access-request procedures, breach handling, and audits. The broader legal framework is the [Data Privacy Act of 2012](https://privacy.gov.ph/data-privacy-act/). Obtain qualified legal/privacy review for the actual operator and deployment; this report is an engineering plan, not legal advice.

Minimum engineering controls:

- least-privilege roles and mandatory authentication;
- HTTPS, secure cookies, CSRF protection, secret management, and encrypted backups;
- no RTSP credentials, phone numbers, or sensitive evidence in logs;
- configurable retention by evidence/report category and a deletion audit record;
- visible CCTV notices and documented purpose/coverage;
- evidence access logs and periodic permission review;
- signed model releases and file checksums;
- backup/restore test, incident-response procedure, and provider outage fallback;
- privacy impact assessment before pilot collection;
- no face recognition or license-plate recognition unless separately justified, authorized, and reviewed.

## 11. Phased implementation backlog

### Phase 0 — Requirements and governance (week 1)

- Confirm operator, sites, dispatcher/responder roles, escalation policy, notification channels, SLAs, and report fields.
- Approve dataset terms and a local-footage collection/retention policy.
- Define success criteria and a privacy/security checklist.

**Done when:** workflow/state diagram, permission matrix, label guide, dataset register, and pilot policy are approved.

### Phase 1 — Human review and audit foundation (weeks 2–3)

- Create `operations` app, review/audit/model-release models, migrations, and permissions.
- Implement claim, confirm, dismiss, and review-history endpoints/forms.
- Add tests for authorization, race conditions, invalid transitions, and audit creation.

**Done when:** two concurrent dispatchers cannot double-confirm an incident, every action is attributable, and raw AI events cannot be marked dispatched.

### Phase 2 — Dispatch and responder workflow (weeks 4–5)

- Add responder profiles, dispatches, assignments, mobile responder page, and status transitions.
- Add SLA timers and dispatcher escalation indicators.
- Test duplicate clicks, stale forms, reassignment, decline, cancellation, and responder access boundaries.

**Done when:** one confirmed test incident can complete the full dispatcher-to-responder-to-resolution lifecycle with a complete audit timeline.

### Phase 3 — Durable notifications and audio (weeks 6–7)

- Add outbox/delivery model and durable worker queue.
- Implement in-app notifications, alert sound opt-in/test/mute, then email.
- Add retry, provider callbacks, idempotency, failure UI, and integration fakes.

**Done when:** restarts do not lose notifications, retries do not duplicate them, delivery is not confused with acknowledgement, and muted/blocked audio has a visual fallback.

### Phase 4 — Reports and analytics (weeks 8–9)

- Build incident-report drafting/finalization/versioning.
- Add search/filtering, print layout, CSV aggregate export, and confidentiality markings.
- Build operations/model/camera dashboards with documented metric definitions.

**Done when:** an auditor can reconstruct an incident solely from the report and audit timeline, and every dashboard number links to its underlying records.

### Phase 5 — Model training and shadow validation (parallel, weeks 2–10)

- Acquire approved CCD/DoTA for pipeline validation and fixed-camera datasets for domain evaluation.
- Label local normal/event footage, train GRU baseline, calibrate thresholds, and register the ONNX release.
- Run a two-to-four-week shadow pilot and review false alarms/misses.

**Done when:** held-out metrics and false alarms per camera-hour meet the approved target, subgroup results are documented, model rollback works, and dispatch remains human-authorized.

### Phase 6 — Pilot hardening (weeks 10–12)

- Move from SQLite to PostgreSQL for concurrent operational use.
- Separate web, capture/inference, and task-worker processes.
- Add monitoring, backups, restore drill, retention jobs, HTTPS, and deployment runbooks.
- Conduct tabletop exercises with synthetic/test incidents only.

**Done when:** security/privacy review passes, outage behavior is documented, recovery is tested, and authorized operators sign off.

## 12. Recommended first implementation slice

The best next coding milestone is **Dispatcher Review MVP**, not another model rewrite:

1. Add an `operations` Django app.
2. Extend review states to include `in_review` and `confirmed` while preserving existing data.
3. Add `IncidentReview` and append-only `AuditEvent`.
4. Seed `Administrators`, `Dispatchers`, `Responders`, and `Auditors` groups through a data migration or idempotent management command.
5. Build a dispatcher queue and review form.
6. Require a reason for dismissal and store it as model feedback.
7. Add permission and concurrent-claim tests.
8. Only after that passes, add `Dispatch` and `DispatchAssignment`.

This slice converts the current incident viewer into a controlled operational workflow while preserving the existing detection pipeline. It also begins collecting the most valuable future training data: verified false-positive and confirmed-event outcomes from the deployment domain.

## 13. Pilot acceptance gates

Do not call the system production-ready until all gates pass:

- **Model:** predefined held-out recall/false-alert target, condition breakdown, model versioning, and rollback.
- **Human control:** no responder notification from an unconfirmed AI event.
- **Reliability:** restart/retry tests, idempotent notifications, backup/restore drill, camera/provider outage behavior.
- **Security/privacy:** least privilege, HTTPS, secrets, access logs, retention, notices, privacy impact assessment, and approved terms.
- **Operations:** trained dispatchers/responders, escalation rules, drills, and manual fallback.
- **Reporting:** immutable timeline, report revisions, traceable exports, and reconciled analytics.

## 14. Immediate decisions required from the project owner

Before Phase 1 is finalized, decide:

1. Is the first deployment an academic demo, a private-site pilot, or an LGU/public-road pilot?
2. Who is legally/operationally permitted to be a dispatcher and responder?
3. Which notification channel is approved for the pilot: in-app only, email, SMS, or an existing agency system?
4. What acknowledgement and escalation times apply by priority?
5. What footage/report retention periods are required?
6. What measurable model target is acceptable (especially false alerts per camera-hour and minimum recall)?

Until those are answered, implementation can safely proceed through the local Dispatcher Review MVP using test incidents, but external responder messaging and real emergency integration should remain disabled.
