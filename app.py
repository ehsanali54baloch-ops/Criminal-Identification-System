#!/usr/bin/env python3
"""
==============================================================================
AI-Based Face Detection & Criminal Identification System
Main Gradio Web Dashboard (app.py)
Team: Ehsan Ali (Leader), Asif Ali, Sabir Ali
==============================================================================
This dashboard provides real-time surveillance monitoring via browser webcam,
live criminal biometric database matching, mask detection, instant alerts,
and a comprehensive suspect management system.
"""

import os
import json
import sqlite3
import numpy as np
import pandas as pd
import gradio as gr
from PIL import Image
import cv2

from face_utils import FaceUtils, DB_PATH, DEEPFACE_AVAILABLE, MODEL_ERROR, EMAIL_COOLDOWN_SEC
import email_service


# ------------------------------------------------------------------------------
# Alarm sound: a 3-second two-tone police siren, generated in code (no file needed).
# It plays in the operator's BROWSER (Codespaces servers have no speakers).
# ------------------------------------------------------------------------------
ALARM_SR = 22050

def make_siren(seconds: float = 3.0) -> np.ndarray:
    t = np.arange(int(ALARM_SR * seconds)) / ALARM_SR
    freq = np.where((t * 2.5).astype(int) % 2 == 0, 960.0, 640.0)   # hi-lo siren
    phase = 2 * np.pi * np.cumsum(freq) / ALARM_SR
    wave = 0.6 * np.sign(np.sin(phase)) * 0.5 + 0.4 * np.sin(phase)
    fade = np.minimum(1.0, np.minimum(t, seconds - t) * 20)          # no clicks
    return (wave * fade * 0.8 * 32767).astype(np.int16)

SIREN = make_siren()


def alarm_output(play: bool):
    """New alert -> fresh audio clip (autoplays). Otherwise leave the player alone."""
    if not play:
        return gr.skip()
    # tiny random dither makes every clip a new file, so the browser plays it again
    return (ALARM_SR, SIREN + np.random.randint(-2, 3, SIREN.shape).astype(np.int16))
from database_setup import init_database, MUGSHOTS_DIR

# Ensure DB is initialized
if not os.path.exists(DB_PATH):
    init_database()


def to_bgr(img) -> np.ndarray:
    """Gradio gives RGB or RGBA numpy arrays (or PIL). Convert to OpenCV BGR."""
    arr = np.array(img)
    if arr.ndim == 2:
        return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    if arr.shape[2] == 4:
        return cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def get_db_connection():
    return sqlite3.connect(DB_PATH)


def load_criminals_table():
    """Fetches all enrolled suspects as a clean Pandas DataFrame."""
    conn = get_db_connection()
    df = pd.read_sql_query("""
        SELECT suspect_id AS "ID", name AS "Full Name", alias AS "Alias",
               age AS "Age", crime_category AS "Crime Category",
               threat_level AS "Threat Level", bounty AS "Bounty",
               status AS "Status", last_seen AS "Last Seen"
        FROM criminals ORDER BY id DESC
    """, conn)
    conn.close()
    return df


def load_incident_logs():
    """Fetches recent security incident alerts from SQLite."""
    conn = get_db_connection()
    df = pd.read_sql_query("""
        SELECT id AS "Alert ID", suspect_id AS "Suspect ID",
               suspect_name AS "Identified Suspect",
               confidence_score AS "Match Confidence (%)",
               CASE WHEN mask_detected = 1 THEN 'MASK DETECTED' ELSE 'NO MASK' END AS "Mask Status",
               threat_level AS "Threat Level",
               location AS "Camera / Location",
               COALESCE(email_status, '-') AS "Admin Email",
               snapshot_path AS "Snapshot",
               alert_timestamp AS "Timestamp"
        FROM incident_alerts ORDER BY id DESC LIMIT 50
    """, conn)
    conn.close()
    return df


def process_surveillance_feed(input_img, threshold: float, enable_mask: bool):
    """
    Core handler invoked whenever a frame arrives from gr.Webcam or gr.Image.
    """
    if input_img is None:
        return (
            None,
            "<div style='padding:12px; background:#1e293b; color:#94a3b8; border-radius:8px;'>Awaiting surveillance camera input...</div>",
            "<div style='padding:12px; background:#0f172a; color:#64748b; border-radius:8px;'>No suspect detected. Feed normal.</div>",
            load_incident_logs(),
            gr.skip()
        )

    # Convert RGB (from Gradio) to BGR (for OpenCV)
    frame_bgr = to_bgr(input_img)

    # Convert percentage threshold to 0.0 - 1.0 range
    thresh_decimal = threshold / 100.0

    # Run complete CV pipeline
    annotated_bgr, detections = FaceUtils.process_frame(
        frame_bgr,
        similarity_threshold=thresh_decimal,
        enable_mask_detection=enable_mask
    )

    # Convert back to RGB for Gradio display
    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

    # Build Alert Banner and Suspect Dossier
    matches = [d for d in detections if d["is_match"]]

    new_alert = any(d.get("new_alert") for d in detections)
    if email_service.is_configured():
        email_line = (f"📧 Snapshot emailed to admin ({email_service.get_settings()['admin']}). "
                      f"Next email for the same person after {EMAIL_COOLDOWN_SEC}s.")
    else:
        email_line = "⚠️ Admin email NOT configured — open the 📧 Email Alert Settings tab."

    if matches:
        suspect = matches[0]["suspect"]
        mask_info = matches[0]["mask_info"]

        # Glowing Red Alert HTML
        alert_html = f"""
        <div style="background: linear-gradient(135deg, #7f1d1d, #991b1b); border: 2px solid #ef4444; 
                    padding: 16px; border-radius: 10px; color: #ffffff; box-shadow: 0 0 20px rgba(239, 68, 68, 0.4);">
            <div style="display:flex; align-items:center; justify-content:space-between;">
                <div style="font-size: 18px; font-weight: 800; letter-spacing: 1px; display:flex; align-items:center; gap:8px;">
                    <span style="font-size:24px;">🚨</span> SECURITY ALERT: CRIMINAL MATCH IDENTIFIED!
                </div>
                <div style="background:#dc2626; padding: 4px 12px; border-radius: 9999px; font-size:12px; font-weight:700;">
                    THREAT: {suspect['threat_level']}
                </div>
            </div>
            <div style="margin-top: 8px; font-size: 14px; opacity: 0.95;">
                Matched suspect <strong>{suspect['name']}</strong> with <strong>{suspect['confidence_score']}% confidence</strong>. 
                <br><span style="font-size:13px;">{email_line}</span>
            </div>
        </div>
        """

        # Detailed Suspect Dossier
        dossier_html = f"""
        <div style="background: #0f172a; border: 1px solid #334155; border-radius: 10px; padding: 20px; color: #e2e8f0;">
            <div style="font-size: 16px; font-weight: 700; color: #38bdf8; margin-bottom: 12px; border-bottom: 1px solid #1e293b; padding-bottom: 8px;">
                SUSPECT BIOMETRIC DOSSIER
            </div>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px; font-size: 13px;">
                <div><span style="color:#94a3b8;">Suspect ID:</span> <strong>{suspect['suspect_id']}</strong></div>
                <div><span style="color:#94a3b8;">Full Name:</span> <strong>{suspect['name']}</strong></div>
                <div><span style="color:#94a3b8;">Alias:</span> <em>{suspect.get('alias', 'N/A')}</em></div>
                <div><span style="color:#94a3b8;">Threat Level:</span> <span style="color:{'#ef4444' if suspect['threat_level']=='HIGH' else '#f59e0b'}; font-weight:700;">{suspect['threat_level']}</span></div>
                <div><span style="color:#94a3b8;">Crime:</span> {suspect['crime_category']}</div>
                <div><span style="color:#94a3b8;">Bounty:</span> <span style="color:#10b981; font-weight:700;">{suspect['bounty']}</span></div>
                <div><span style="color:#94a3b8;">Status:</span> {suspect['status']}</div>
                <div><span style="color:#94a3b8;">Mask Verification:</span> <span style="color:{'#38bdf8' if mask_info['is_masked'] else '#94a3b8'};">{mask_info['status']}</span></div>
            </div>
        </div>
        """
    else:
        alert_html = """
        <div style="background: #064e3b; border: 1px solid #059669; padding: 14px; border-radius: 10px; color: #ecfdf5;">
            <div style="font-weight: 700; font-size: 15px;">🛡️ SURVEILLANCE ACTIVE — SECTOR CLEAR</div>
            <div style="font-size: 13px; opacity: 0.9; margin-top: 4px;">Faces tracked in real-time. No matches exceeding the confidence threshold found.</div>
        </div>
        """
        
        face_count = len(detections)
        sims = [d["best_similarity"] for d in detections if d.get("best_similarity") is not None]
        closest = f"{max(sims)}%" if sims else "—"
        model_state = "FaceNet512 ACTIVE" if DEEPFACE_AVAILABLE else f"NOT LOADED ({MODEL_ERROR[:80]})"
        dossier_html = f"""
        <div style="background: #0f172a; border: 1px solid #1e293b; border-radius: 10px; padding: 16px; color: #94a3b8; font-size: 13px;">
            <div style="color: #64748b; font-weight: 600; margin-bottom: 6px;">MONITORING STATUS</div>
            <div>Active Faces Tracked: <strong>{face_count}</strong></div>
            <div>Recognition Model: <strong>{model_state}</strong></div>
            <div>Suspects with face data: <strong>{FaceUtils.enrolled_count()}</strong></div>
            <div>Closest similarity in frame: <strong>{closest}</strong> (threshold {threshold:.0f}%)</div>
            <div>Mask Analyzer: <strong>{'Enabled' if enable_mask else 'Disabled'}</strong></div>
        </div>
        """

    return annotated_rgb, alert_html, dossier_html, load_incident_logs(), alarm_output(new_alert)


def enroll_new_suspect(name, alias, age, gender, crime, threat, bounty, status, last_seen, image_input):
    """
    Enrolls a new criminal record into SQLite database with extracted embedding.
    """
    if not name or image_input is None:
        return "❌ Error: Both 'Full Name' and 'Suspect Image' are strictly required.", load_criminals_table()

    try:
        img_bgr = to_bgr(image_input)
        embedding, msg = FaceUtils.enrollment_embedding(img_bgr)
        if embedding is None:
            return f"❌ Enrollment failed: {msg}", load_criminals_table()
        emb_json = json.dumps([embedding.tolist()])   # list of photos (1 for now)

        # Safe unique ID (COUNT(*) breaks after a delete)
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT COALESCE(MAX(id), 0) FROM criminals")
        next_num = cursor.fetchone()[0] + 1
        suspect_id = f"CRIM-{8000 + next_num}"

        os.makedirs(MUGSHOTS_DIR, exist_ok=True)
        safe_name = "".join(c for c in name if c.isalnum() or c in (' ', '_')).rstrip().replace(' ', '_').lower()
        save_path = os.path.join(MUGSHOTS_DIR, f"{safe_name}_{next_num}.jpg")
        cv2.imwrite(save_path, img_bgr)

        # Insert to SQLite
        cursor.execute("""
            INSERT INTO criminals (
                suspect_id, name, alias, age, gender, crime_category,
                threat_level, bounty, status, last_seen, image_path, face_embedding
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            suspect_id, name, alias, int(age or 30), gender, crime,
            threat, bounty or "$0", status, last_seen or "Unknown",
            save_path, emb_json
        ))
        conn.commit()
        conn.close()
        FaceUtils.invalidate_cache()

        return f"✅ Successfully enrolled [{suspect_id}] {name} into Biometric Database!", load_criminals_table()
    except Exception as e:
        return f"❌ Database Insertion Error: {str(e)}", load_criminals_table()


def delete_suspect(suspect_id: str):
    """Removes a suspect record by ID."""
    if not suspect_id:
        return "Please enter a valid Suspect ID to delete.", load_criminals_table()
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM criminals WHERE suspect_id = ?", (suspect_id.strip(),))
    deleted = cursor.rowcount
    conn.commit()
    conn.close()
    FaceUtils.invalidate_cache()
    if deleted > 0:
        return f"✅ Deleted record {suspect_id} from database.", load_criminals_table()
    return f"⚠️ Suspect ID '{suspect_id}' not found.", load_criminals_table()


def add_photo_to_suspect(suspect_id: str, image_input):
    """
    Adds one more face photo to an existing suspect. More photos
    (different angles / lighting) = much better recognition.
    """
    if not suspect_id or image_input is None:
        return "❌ Enter a Suspect ID and choose a photo.", load_criminals_table()
    embedding, msg = FaceUtils.enrollment_embedding(to_bgr(image_input))
    if embedding is None:
        return f"❌ {msg}", load_criminals_table()
    conn = get_db_connection()
    row = conn.execute("SELECT face_embedding FROM criminals WHERE suspect_id = ?",
                       (suspect_id.strip(),)).fetchone()
    if row is None:
        conn.close()
        return f"⚠️ Suspect ID '{suspect_id}' not found.", load_criminals_table()
    vectors = []
    try:
        data = json.loads(row[0]) if row[0] else []
        vectors = data if (data and isinstance(data[0], list)) else ([data] if data else [])
        vectors = [v for v in vectors if len(v) == len(embedding)]   # drop old 128-d data
    except Exception:
        vectors = []
    vectors.append(embedding.tolist())
    conn.execute("UPDATE criminals SET face_embedding = ? WHERE suspect_id = ?",
                 (json.dumps(vectors), suspect_id.strip()))
    conn.commit()
    conn.close()
    FaceUtils.invalidate_cache()
    return f"✅ Added photo. {suspect_id} now has {len(vectors)} face photo(s).", load_criminals_table()


def save_email_settings(admin_email, smtp_user, smtp_password, smtp_host, smtp_port):
    """Saves settings for this session and sends a test email."""
    email_service.configure(admin_email, smtp_user, smtp_password, smtp_host, smtp_port)
    if not email_service.is_configured():
        return "❌ Please fill Admin Email, Sender Email and App Password."
    ok, info = email_service.send_test_email()
    if ok:
        return f"✅ Test email sent! {info}. Alerts with snapshots will now be emailed automatically."
    return f"❌ Could not send: {info}"


def email_status_text():
    s = email_service.get_settings()
    if email_service.is_configured():
        return f"Configured: alerts go to {s['admin']} (sender {s['user']} via {s['host']}:{s['port']})"
    return "Not configured yet."


# ==============================================================================
# Gradio UI Construction
# ==============================================================================

custom_theme = gr.themes.Soft(
    primary_hue="red",
    secondary_hue="slate",
    neutral_hue="slate"
).set(
    body_background_fill="#090d16",
    block_background_fill="#111827",
    block_border_width="1px",
    block_border_color="#1f2937"
)

with gr.Blocks(theme=custom_theme, title="AI Face Detection & Criminal Identification System") as demo:
    gr.HTML("""
    <div style="text-align: center; padding: 24px 0 16px 0; border-bottom: 1px solid #1f2937; margin-bottom: 20px;">
        <div style="display:inline-block; padding: 4px 14px; background: rgba(239, 68, 68, 0.15); border: 1px solid #ef4444; border-radius: 9999px; color: #f87171; font-size: 12px; font-weight: 700; letter-spacing: 1.5px; margin-bottom: 8px;">
            LAW ENFORCEMENT & SECURITY INTELLIGENCE
        </div>
        <h1 style="color: #f8fafc; font-size: 28px; font-weight: 800; margin: 0; letter-spacing: -0.5px;">
            AI-Based Face Detection & Criminal Identification System
        </h1>
        <p style="color: #94a3b8; font-size: 14px; margin-top: 6px;">
            Real-Time Edge Biometric Surveillance • FaceNet Recognition • Mask Detection • SQLite Core
        </p>
        <div style="font-size: 12px; color: #64748b; margin-top: 4px;">
            Developed by: <strong>Ehsan Ali (Leader)</strong>, <strong>Asif Ali</strong>, <strong>Sabir Ali</strong>
        </div>
    </div>
    """)

    with gr.Tabs():
        # ----------------------------------------------------------------------
        # Tab 1: Live Surveillance & Detection
        # ----------------------------------------------------------------------
        with gr.TabItem("🎥 Live Surveillance & Detection"):
            with gr.Row():
                with gr.Column(scale=5):
                    with gr.Tabs():
                        with gr.TabItem("Webcam Stream"):
                            webcam_feed = gr.Image(
                                label="Operator Live Webcam Stream",
                                sources=["webcam"],
                                streaming=True,
                                type="numpy"
                            )
                        with gr.TabItem("Upload Still Image / CCTV Snapshot"):
                            image_feed = gr.Image(
                                label="Upload Investigation Photo",
                                type="numpy"
                            )

                    with gr.Row():
                        conf_slider = gr.Slider(
                            minimum=50, maximum=95, value=70, step=1,
                            label="Match Similarity Threshold (%)",
                            info="FaceNet512 cosine similarity. ~70% works well; raise it if you get false alarms."
                        )
                        mask_toggle = gr.Checkbox(
                            value=True,
                            label="Enable Mask Verification",
                            info="Detects protective medical/cloth masks on subjects."
                        )

                with gr.Column(scale=6):
                    alert_banner = gr.HTML("""
                        <div style="background: #064e3b; border: 1px solid #059669; padding: 14px; border-radius: 10px; color: #ecfdf5;">
                            <div style="font-weight: 700; font-size: 15px;">🛡️ SURVEILLANCE READY</div>
                            <div style="font-size: 13px; opacity: 0.9;">System standing by. Click camera access to start surveillance stream.</div>
                        </div>
                    """)

                    alarm_audio = gr.Audio(
                        label="🔊 Alarm (plays automatically on a new match — keep this tab's sound on)",
                        autoplay=True,
                        interactive=False,
                        type="numpy"
                    )

                    annotated_output = gr.Image(
                        label="Tactical Cyber HUD Analysis Feed",
                        type="numpy"
                    )

                    suspect_dossier = gr.HTML("""
                        <div style="background: #0f172a; border: 1px solid #1e293b; border-radius: 10px; padding: 16px; color: #94a3b8; font-size: 13px;">
                            <div style="color: #64748b; font-weight: 600; margin-bottom: 6px;">DOSSIER VIEWER</div>
                            <div>No suspect in view. Biometric recognition active.</div>
                        </div>
                    """)

            gr.HTML("<h3 style='color:#e2e8f0; margin-top:20px;'>🚨 Live Incident Alert Logs</h3>")
            live_logs_table = gr.DataFrame(
                value=load_incident_logs(),
                interactive=False
            )

            # Bind webcam stream event
            webcam_feed.stream(
                fn=process_surveillance_feed,
                inputs=[webcam_feed, conf_slider, mask_toggle],
                outputs=[annotated_output, alert_banner, suspect_dossier, live_logs_table, alarm_audio],
                stream_every=0.5,        # ~2 frames/sec: FaceNet on CPU needs time
                concurrency_limit=1,     # drop frames instead of building a queue
                show_progress="hidden"
            )

            # Bind static image change event
            image_feed.change(
                fn=process_surveillance_feed,
                inputs=[image_feed, conf_slider, mask_toggle],
                outputs=[annotated_output, alert_banner, suspect_dossier, live_logs_table, alarm_audio]
            )

        # ----------------------------------------------------------------------
        # Tab 2: Criminal Database Management
        # ----------------------------------------------------------------------
        with gr.TabItem("🗄️ Criminal Database Management"):
            with gr.Row():
                with gr.Column(scale=6):
                    gr.HTML("<h3 style='color:#e2e8f0;'>Active Suspect Registry (SQLite)</h3>")
                    criminals_table = gr.DataFrame(
                        value=load_criminals_table(),
                        interactive=False
                    )
                    refresh_btn = gr.Button("🔄 Refresh Database View")
                    
                    with gr.Row():
                        del_id_input = gr.Textbox(label="Delete Record by Suspect ID", placeholder="e.g. CRIM-8921")
                        del_btn = gr.Button("🗑️ Delete Suspect", variant="stop")
                    del_status = gr.Textbox(label="Operation Status", interactive=False)

                    gr.HTML("<h3 style='color:#e2e8f0; margin-top:16px;'>Add Another Photo to a Suspect</h3>"
                            "<div style='color:#94a3b8; font-size:13px;'>3–5 photos per person (different angles and lighting) "
                            "makes recognition far more reliable.</div>")
                    with gr.Row():
                        extra_id = gr.Textbox(label="Suspect ID", placeholder="e.g. CRIM-8004")
                        extra_img = gr.Image(label="Extra Face Photo", type="numpy", sources=["upload", "webcam"])
                    extra_btn = gr.Button("➕ Add Photo")
                    extra_status = gr.Textbox(label="Result", interactive=False)

                with gr.Column(scale=5):
                    gr.HTML("<h3 style='color:#e2e8f0;'>Enroll New Suspect Profile</h3>")
                    with gr.Group():
                        new_name = gr.Textbox(label="Full Name *", placeholder="e.g. John Doe")
                        new_alias = gr.Textbox(label="Alias / Known Moniker", placeholder="e.g. The Fox")
                        with gr.Row():
                            new_age = gr.Number(label="Age", value=32)
                            new_gender = gr.Dropdown(choices=["Male", "Female", "Unknown"], value="Male", label="Gender")
                        new_crime = gr.Textbox(label="Crime Category / Allegations", placeholder="e.g. Armed Robbery")
                        with gr.Row():
                            new_threat = gr.Dropdown(choices=["HIGH", "MEDIUM", "LOW"], value="HIGH", label="Threat Level")
                            new_bounty = gr.Textbox(label="Bounty Reward", value="$50,000")
                        with gr.Row():
                            new_status = gr.Dropdown(choices=["WANTED", "SURVEILLANCE", "IN_CUSTODY"], value="WANTED", label="Status")
                            new_last_seen = gr.Textbox(label="Last Known Location", placeholder="e.g. Terminal 2B")
                        
                        new_image = gr.Image(label="Suspect Mugshot Photo *", type="numpy", sources=["upload", "webcam"])
                        enroll_btn = gr.Button("💾 Extract Biometrics & Enroll into Database", variant="primary")
                        enroll_status = gr.Textbox(label="Enrollment Result", interactive=False)

            # Event listeners
            refresh_btn.click(fn=load_criminals_table, outputs=criminals_table)
            enroll_btn.click(
                fn=enroll_new_suspect,
                inputs=[new_name, new_alias, new_age, new_gender, new_crime, new_threat, new_bounty, new_status, new_last_seen, new_image],
                outputs=[enroll_status, criminals_table]
            )
            del_btn.click(
                fn=delete_suspect,
                inputs=[del_id_input],
                outputs=[del_status, criminals_table]
            )
            extra_btn.click(
                fn=add_photo_to_suspect,
                inputs=[extra_id, extra_img],
                outputs=[extra_status, criminals_table]
            )

        # ----------------------------------------------------------------------
        # Tab: Email Alert Settings
        # ----------------------------------------------------------------------
        with gr.TabItem("📧 Email Alert Settings"):
            gr.Markdown("""
            ### Send alert snapshots to the admin
            When a criminal is confirmed, the system saves a snapshot and emails it to the admin.

            **Using Gmail as the sender:** turn on *2-Step Verification* in your Google account,
            then create an **App Password** (Google Account → Security → App passwords) and paste
            the 16-letter code below. Your normal Gmail password will **not** work.

            Settings typed here last until the app restarts. To keep them permanently, put them
            in a `.env` file (see README) — never commit that file to GitHub.
            """)
            email_state = gr.Textbox(label="Current status", value=email_status_text(), interactive=False)
            with gr.Row():
                cfg_admin = gr.Textbox(label="Admin Email (receives alerts) *",
                                       value=email_service.get_settings()["admin"],
                                       placeholder="admin@example.com")
                cfg_user = gr.Textbox(label="Sender Email (Gmail account) *",
                                      value=email_service.get_settings()["user"],
                                      placeholder="yourproject@gmail.com")
            with gr.Row():
                cfg_pass = gr.Textbox(label="App Password *", type="password",
                                      placeholder="16-letter Gmail App Password")
                cfg_host = gr.Textbox(label="SMTP Server", value=email_service.get_settings()["host"])
                cfg_port = gr.Number(label="SMTP Port", value=email_service.get_settings()["port"], precision=0)
            cfg_btn = gr.Button("💾 Save & Send Test Email", variant="primary")
            cfg_result = gr.Textbox(label="Result", interactive=False)
            cfg_btn.click(
                fn=save_email_settings,
                inputs=[cfg_admin, cfg_user, cfg_pass, cfg_host, cfg_port],
                outputs=[cfg_result]
            ).then(fn=email_status_text, outputs=[email_state])

        # ----------------------------------------------------------------------
        # Tab 3: System Pipeline & Architecture
        # ----------------------------------------------------------------------
        with gr.TabItem("⚙️ Pipeline & System Architecture"):
            gr.Markdown("""
            ### 🔍 Five-Stage Surveillance Pipeline
            
            ```
            [ 1. CAPTURE ] ---> [ 2. DETECT ] ---> [ 3. EXTRACT ] ---> [ 4. COMPARE ] ---> [ 5. ALERT ]
                 │                    │                   │                   │                  │
            Webcam Frame        DeepFace detector     FaceNet512         Cosine Similarity    Console Log /
            640x480 RGB         + eye alignment     512-d Feature Vec     against SQLite      Red HUD Flash
                                        │
                                  [ MASK DETECTOR ]
                                HSV & Edge Variance
                                                                                         │
                                                              [ ALARM SOUND + SNAPSHOT + EMAIL TO ADMIN ]
            ```

            #### Core System Specifications:
            * **Computer Vision Engine:** OpenCV (`opencv-python-headless`) for containerized deployment.
            * **Facial Representation:** DeepFace / FaceNet512 — 512-dimensional face embeddings (multiple photos per suspect).
            * **Database Engine:** Embedded SQLite3 (`criminals.db`) with zero external dependency overhead.
            * **Mask Classification:** Multi-channel color space and texture edge density ratio.
            * **Development Team:** **Ehsan Ali (Leader)**, **Asif Ali**, **Sabir Ali**.
            """)

# Launch configuration (codespaces / cloud friendly)
if __name__ == "__main__":
    # Server binds to 0.0.0.0 for container networking
    demo.queue().launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )
