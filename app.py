from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import os
import mediapipe as mp
import cv2
import numpy as np
from werkzeug.security import generate_password_hash, check_password_hash
from base64 import b64decode
import pytz
from PIL import Image
import io
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from flask import send_file
from flask_migrate import Migrate
import jwt
from datetime import datetime, timedelta
from functools import wraps
from scipy.spatial.distance import cosine
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT

mp_face_mesh = mp.solutions.face_mesh
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

app = Flask(__name__)

app.config["SECRET_KEY"] = os.urandom(24)
app.config["SQLALCHEMY_DATABASE_URI"] = "postgresql://postgres:8100@localhost/face"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["TIMEZONE"] = "Asia/Kolkata"
app.config["JWT_SECRET_KEY"] = (
    "your_jwt_secret_key"  # Change this to a strong secret key
)
app.config["JWT_ACCESS_TOKEN_EXPIRES"] = timedelta(hours=1)

db = SQLAlchemy(app)
migrate = Migrate(app, db)


# Database Models
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(200))
    face_embedding = db.Column(db.PickleType)
    team = db.Column(db.String(100))
    role = db.Column(db.String(100))
    registered_at = db.Column(
        db.DateTime, default=lambda: datetime.now(pytz.timezone("Asia/Kolkata"))
    )
    profile_image = db.Column(db.LargeBinary)
    phone = db.Column(db.String(20))


class Attendance(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    timestamp = db.Column(
        db.DateTime, default=lambda: datetime.now(pytz.timezone("Asia/Kolkata"))
    )
    type = db.Column(db.String(3))  # 'IN' or 'OUT'
    latitude = db.Column(db.Float)
    longitude = db.Column(db.Float)
    address = db.Column(db.String(500))  # Store geocoded address
    device_info = db.Column(db.String(200))  # Store device information


class LocationTrack(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    timestamp = db.Column(
        db.DateTime, default=lambda: datetime.now(pytz.timezone("Asia/Kolkata"))
    )
    latitude = db.Column(db.Float)
    longitude = db.Column(db.Float)
    address = db.Column(db.String(500))


# JWT Authentication Functions
def create_access_token(user_id):
    payload = {
        "user_id": user_id,
        "exp": datetime.utcnow() + app.config["JWT_ACCESS_TOKEN_EXPIRES"],
    }
    return jwt.encode(payload, app.config["JWT_SECRET_KEY"], algorithm="HS256")


def get_user_from_token(token):
    try:
        payload = jwt.decode(token, app.config["JWT_SECRET_KEY"], algorithms=["HS256"])
        return User.query.get(payload["user_id"])
    except:
        return None


def jwt_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        token = request.headers.get("Authorization", None)
        if not token:
            return jsonify({"error": "Missing token"}), 401

        try:
            token = token.split(" ")[1]
            user = get_user_from_token(token)
            if not user:
                return jsonify({"error": "Invalid token"}), 401
        except jwt.ExpiredSignatureError:
            return jsonify({"error": "Token has expired"}), 401
        except jwt.InvalidTokenError:
            return jsonify({"error": "Invalid token"}), 401
        except Exception as e:
            return jsonify({"error": str(e)}), 401

        return f(user=user, *args, **kwargs)

    return decorated_function


def generate_pdf_report(user_id, start_date, end_date):
    try:
        # Convert string dates to datetime objects
        if isinstance(start_date, str):
            start_date = datetime.strptime(start_date, "%Y-%m-%d")
        if isinstance(end_date, str):
            end_date = datetime.strptime(end_date, "%Y-%m-%d")

        # Add time to end_date to include the whole day
        end_date = end_date.replace(hour=23, minute=59, second=59)

        user = User.query.get(user_id)
        if not user:
            raise ValueError("User not found")

        attendances = (
            Attendance.query.filter(
                Attendance.user_id == user_id,
                Attendance.timestamp.between(start_date, end_date),
            )
            .order_by(Attendance.timestamp)
            .all()
        )

        buffer = io.BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=letter,
            rightMargin=inch / 2,
            leftMargin=inch / 2,
            topMargin=inch / 2,
            bottomMargin=inch / 2,
        )

        styles = getSampleStyleSheet()
        elements = []

        # Custom styles
        styles.add(
            ParagraphStyle(
                name="ReportTitle", fontSize=16, alignment=TA_CENTER, spaceAfter=20
            )
        )

        styles.add(
            ParagraphStyle(
                name="SectionHeader",
                fontSize=12,
                textColor=colors.HexColor("#2E75B6"),
                spaceAfter=10,
            )
        )

        styles.add(
            ParagraphStyle(
                name="SummaryText",
                fontSize=10,
                textColor=colors.HexColor("#4472C4"),
                spaceAfter=5,
            )
        )

        # Add report title
        title = f"ATTENDANCE REPORT - {user.name.upper()}"
        elements.append(Paragraph(title, styles["ReportTitle"]))

        # Add date range
        date_range = f"Period: {start_date.strftime('%d %b %Y')} to {end_date.strftime('%d %b %Y')}"
        elements.append(Paragraph(date_range, styles["SectionHeader"]))
        elements.append(Spacer(1, 20))

        if not attendances:
            elements.append(
                Paragraph(
                    "No attendance records found for this period", styles["BodyText"]
                )
            )
        else:
            # Prepare data for detailed entries table
            detailed_data = [["Date", "Time", "Type", "Location"]]
            daily_summary = {}
            total_working_seconds = 0

            for attendance in attendances:
                date_str = attendance.timestamp.strftime("%Y-%m-%d")
                time_str = attendance.timestamp.strftime("%H:%M:%S")

                detailed_data.append(
                    [date_str, time_str, attendance.type, attendance.address or "N/A"]
                )

                # Calculate working hours for summary
                if date_str not in daily_summary:
                    daily_summary[date_str] = {"check_in": None, "check_out": None}

                if attendance.type == "IN":
                    daily_summary[date_str]["check_in"] = attendance.timestamp
                elif attendance.type == "OUT":
                    daily_summary[date_str]["check_out"] = attendance.timestamp
                    if daily_summary[date_str]["check_in"]:
                        duration = (
                            attendance.timestamp - daily_summary[date_str]["check_in"]
                        ).total_seconds()
                        total_working_seconds += duration
                        daily_summary[date_str]["duration"] = duration

            # Detailed Attendance Records
            elements.append(
                Paragraph("Detailed Attendance Records", styles["SectionHeader"])
            )

            detailed_table = Table(detailed_data)
            detailed_table.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2E75B6")),
                        ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                        ("FONTSIZE", (0, 0), (-1, 0), 10),
                        ("BOTTOMPADDING", (0, 0), (-1, 0), 6),
                        ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#D9E1F2")),
                        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#7F7F7F")),
                        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ]
                )
            )

            elements.append(detailed_table)
            elements.append(Spacer(1, 20))

            # Daily Summary Section (Simplified - Only Date and Working Hours)
            elements.append(
                Paragraph("Daily Working Hours Summary", styles["SectionHeader"])
            )

            summary_data = [["Date", "Working Hours"]]
            for date, records in sorted(daily_summary.items()):
                duration = records.get("duration", 0)
                working_hours = f"{duration/3600:.2f} hours" if duration else "N/A"
                summary_data.append([date, working_hours])

            summary_table = Table(summary_data)
            summary_table.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#70AD47")),
                        ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                        ("FONTSIZE", (0, 0), (-1, 0), 10),
                        ("BOTTOMPADDING", (0, 0), (-1, 0), 6),
                        ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#E2EFDA")),
                        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#7F7F7F")),
                    ]
                )
            )

            elements.append(summary_table)
            elements.append(Spacer(1, 20))

            # Final Summary
            total_days = len(daily_summary)
            total_working_hours = total_working_seconds / 3600
            avg_hours_per_day = (
                total_working_hours / total_days if total_days > 0 else 0
            )

            summary_text = [
                Paragraph("REPORT SUMMARY", styles["SectionHeader"]),
                Spacer(1, 10),
                Paragraph(f"Total Days: {total_days}", styles["SummaryText"]),
                Paragraph(
                    f"Total Working Hours: {total_working_hours:.2f} hours",
                    styles["SummaryText"],
                ),
                Paragraph(
                    f"Average Hours Per Day: {avg_hours_per_day:.2f} hours",
                    styles["SummaryText"],
                ),
            ]

            elements.extend(summary_text)

        # Add footer
        elements.append(Spacer(1, 20))
        elements.append(
            Paragraph(
                f"Generated on {datetime.now().strftime('%d %b %Y %H:%M:%S')}",
                styles["Italic"],
            )
        )

        doc.build(elements)
        buffer.seek(0)
        return buffer

    except Exception as e:
        print(f"Error generating PDF: {str(e)}")
        raise


def detect_liveness(img):
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        min_detection_confidence=0.5, min_tracking_confidence=0.5
    )
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = face_mesh.process(rgb_img)

    if not results.multi_face_landmarks or len(results.multi_face_landmarks) != 1:
        return False

    face_landmarks = results.multi_face_landmarks[0]

    return True


def extract_face_embedding(image_array):
    """Extract face embedding using MediaPipe Face Mesh"""
    with mp_face_mesh.FaceMesh(
        static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5
    ) as face_mesh:

        # Convert to RGB
        image_rgb = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image_rgb)

        if not results.multi_face_landmarks:
            return None  # No face detected

        # Extract face landmarks and convert to embedding
        face_landmarks = results.multi_face_landmarks[0]
        embedding = np.array(
            [
                [landmark.x, landmark.y, landmark.z]
                for landmark in face_landmarks.landmark
            ]
        ).flatten()

        # Normalize embedding
        norm = np.linalg.norm(embedding)
        if norm == 0:
            return None  # Avoid division by zero

        return embedding / norm  # Return normalized embedding


def verify_face(stored_embedding, current_embedding, threshold=0.25):
    """Compare face embeddings using Cosine similarity"""
    if stored_embedding is None or current_embedding is None:
        return False

    similarity = 1 - cosine(stored_embedding, current_embedding)
    print(f"Face Similarity: {similarity:.4f}")

    return similarity > threshold  # Higher similarity means faces match


@app.route("/login", methods=["POST"])
def login():
    try:
        data = request.get_json()
        user = User.query.filter_by(email=data["email"]).first()

        if user and check_password_hash(user.password_hash, data["password"]):
            # Verify face if photo provided
            if "photo" in data:
                image_data = b64decode(data["photo"].split(",")[1])
                nparr = np.frombuffer(image_data, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                current_embedding = extract_face_embedding(img)
                stored_embedding = np.array(user.face_embedding)

                if not verify_face(stored_embedding, current_embedding):
                    return jsonify({"error": "Face verification failed"}), 401

            access_token = create_access_token(user.id)
            return jsonify(
                {
                    "message": "Login successful",
                    "access_token": access_token,
                    "user_id": user.id,
                    "name": user.name,
                    "email": user.email,
                    "role": user.role,
                    "team": user.team,
                }
            )

        return jsonify({"error": "Invalid email or password"}), 401
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/dashboard", methods=["GET"])
@jwt_required
def dashboard(user):
    local_tz = pytz.timezone(app.config["TIMEZONE"])
    today = datetime.now(local_tz).date()

    # Get today's attendance records
    today_attendances = (
        Attendance.query.filter(
            Attendance.user_id == user.id,
            db.func.date(Attendance.timestamp) == today,
        )
        .order_by(Attendance.timestamp.desc())
        .all()
    )

    # Calculate working hours
    total_minutes = 0
    check_ins = [a for a in today_attendances if a.type == "IN"]
    check_outs = [a for a in today_attendances if a.type == "OUT"]

    for check_in, check_out in zip(check_ins, check_outs):
        duration = check_out.timestamp - check_in.timestamp
        total_minutes += duration.total_seconds() / 60

    working_hours = f"{int(total_minutes // 60)}h {int(total_minutes % 60)}m"

    return jsonify(
        {
            "today_attendances": [
                {"type": a.type, "timestamp": a.timestamp, "location": a.address}
                for a in today_attendances
            ],
            "working_hours": working_hours,
        }
    )


@app.route("/mark-attendance", methods=["POST"])
@jwt_required
def mark_attendance(user):
    try:
        data = request.get_json()

        # Process the face image
        image_data = b64decode(data["photo"].split(",")[1])
        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Check liveness
        if not detect_liveness(img):
            return jsonify({"error": "Liveness detection failed"}), 401

        # Verify face
        current_embedding = extract_face_embedding(img)
        stored_embedding = np.array(user.face_embedding)

        if not verify_face(stored_embedding, current_embedding):
            return jsonify({"error": "Face verification failed"}), 401

        # Determine attendance type
        last_attendance = (
            Attendance.query.filter_by(user_id=user.id)
            .order_by(Attendance.timestamp.desc())
            .first()
        )

        attendance_type = (
            "OUT" if last_attendance and last_attendance.type == "IN" else "IN"
        )

        # Create attendance record
        attendance = Attendance(
            user_id=user.id,
            type=attendance_type,
            latitude=data["location"]["latitude"],
            longitude=data["location"]["longitude"],
            address=data["location"].get("address", ""),
            device_info=request.headers.get("User-Agent", ""),
        )

        db.session.add(attendance)
        db.session.commit()

        return jsonify(
            {
                "message": f"Attendance marked as {attendance_type}",
                "attendance": {
                    "type": attendance_type,
                    "timestamp": attendance.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                    "location": {
                        "latitude": attendance.latitude,
                        "longitude": attendance.longitude,
                        "address": attendance.address,
                    },
                },
            }
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/logout", methods=["POST"])
@jwt_required
def logout(user):
    # With JWT, logout is handled client-side by discarding the token
    return jsonify(
        {"message": "Logout successful.", "user_id": user.id, "name": user.name}
    )


@app.route("/register", methods=["POST"])
def register():
    try:
        data = request.get_json()

        # Validate required fields
        required_fields = ["name", "email", "password", "photo"]
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing {field}"}), 400

        if User.query.filter_by(email=data["email"]).first():
            return jsonify({"error": "Email already registered"}), 400

        # Process the face image
        try:
            image_data = b64decode(data["photo"].split(",")[1])
            nparr = np.frombuffer(image_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        except Exception as e:
            return jsonify({"error": f"Invalid image data: {str(e)}"}), 400

        # Get face embedding
        face_embedding = extract_face_embedding(img)
        if face_embedding is None:
            return jsonify({"error": "No face detected in image"}), 400

        # Create new user
        user = User(
            name=data["name"],
            email=data["email"],
            password_hash=generate_password_hash(data["password"]),
            face_embedding=face_embedding.tolist(),
            team=data.get("team"),
            role=data.get("role"),
            profile_image=image_data,
            phone=data.get("phone"),
        )

        db.session.add(user)
        db.session.commit()

        return (
            jsonify(
                {
                    "message": "Registration successful",
                    "user_id": user.id,
                    "name": user.name,
                    "email": user.email,
                    "team": user.team,
                    "role": user.role,
                    "phone": user.phone,
                }
            ),
            201,
        )
    except Exception as e:
        db.session.rollback()
        return jsonify({"error": str(e)}), 500


@app.route("/track-location", methods=["POST"])
@jwt_required
def track_location(user):
    data = request.get_json()

    location = LocationTrack(
        user_id=user.id,
        latitude=data["latitude"],
        longitude=data["longitude"],
        address=data.get("address", ""),
    )

    db.session.add(location)
    db.session.commit()

    return jsonify({"message": "Location tracked successfully"})


@app.route("/generate-report", methods=["POST"])
@jwt_required
def generate_report(user):
    try:
        data = request.get_json()

        # Get parameters from request body
        user_id = data.get(
            "user_id", user.id
        )  # Default to current user if not specified
        start_date = data.get("start_date", datetime.now().strftime("%Y-%m-%d"))
        end_date = data.get("end_date", datetime.now().strftime("%Y-%m-%d"))

        # Validate user access (admin can generate for others)
        if user_id != user.id and user.role != "admin":
            return (
                jsonify({"error": "Unauthorized to generate report for this user"}),
                403,
            )

        pdf_buffer = generate_pdf_report(user_id, start_date, end_date)

        return send_file(
            pdf_buffer,
            download_name=f"attendance_report_{start_date}_to_{end_date}.pdf",
            as_attachment=True,
            mimetype="application/pdf",
        )
    except Exception as e:
        return jsonify({"error": f"Failed to generate report: {str(e)}"}), 500


if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(debug=True)
