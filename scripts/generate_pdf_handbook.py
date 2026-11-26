# scripts/generate_pdf_handbook.py
"""
Generates a publication-quality, professional PDF handbook for the AgriShield AI project.
Contains complete project information, technical foundations, empirical metrics,
member presentation scripts, and top 12 jury Q&A.
"""

import os
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether, HRFlowable
)
from reportlab.pdfgen import canvas

BASE_DIR = r"C:\Users\Admin\.gemini\antigravity\scratch\crop_stress_system"
OUTPUT_PDF = os.path.join(BASE_DIR, "AgriShield_AI_Master_Presentation_Handbook.pdf")

class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_number(num_pages)
            super().showPage()
        super().save()

    def draw_page_number(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#64748B"))
        
        # Header on pages after cover
        if self._pageNumber > 1:
            self.setStrokeColor(colors.HexColor("#E2E8F0"))
            self.setLineWidth(0.75)
            self.line(40, 755, 572, 755)
            self.drawString(40, 760, "AgriShield AI — College Project Defense & Presentation Handbook")
            self.drawRightString(572, 760, "Department of IT & Computer Engineering")

        # Footer
        self.setStrokeColor(colors.HexColor("#E2E8F0"))
        self.setLineWidth(0.75)
        self.line(40, 42, 572, 42)
        self.drawString(40, 30, "AgriShield AI • Precision Multimodal Foliar Pathology & Alert System")
        page_text = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(572, 30, page_text)
        self.restoreState()

def build_pdf():
    doc = SimpleDocTemplate(
        OUTPUT_PDF,
        pagesize=letter,
        leftMargin=40,
        rightMargin=40,
        topMargin=48,
        bottomMargin=50
    )

    styles = getSampleStyleSheet()

    # Custom Palette
    C_PRIMARY = colors.HexColor("#064E3B")    # Dark Green
    C_ACCENT = colors.HexColor("#059669")     # Emerald Accent
    C_TEXT = colors.HexColor("#0F172A")       # Slate Dark
    C_MUTED = colors.HexColor("#475569")      # Slate Muted
    C_BG_CARD = colors.HexColor("#F8FAFC")    # Slate 50
    C_BORDER = colors.HexColor("#CBD5E1")     # Border
    C_GREEN_BG = colors.HexColor("#ECFDF5")   # Soft Green
    C_AMBER_BG = colors.HexColor("#FEF3C7")   # Soft Amber
    C_ROSE_BG = colors.HexColor("#FFE4E6")    # Soft Rose

    # Paragraph Styles
    style_cover_title = ParagraphStyle(
        'CoverTitle', parent=styles['Normal'],
        fontName='Helvetica-Bold', fontSize=22, leading=26,
        textColor=C_PRIMARY, alignment=1, spaceAfter=6
    )
    style_cover_sub = ParagraphStyle(
        'CoverSub', parent=styles['Normal'],
        fontName='Helvetica', fontSize=11, leading=15,
        textColor=C_ACCENT, alignment=1, spaceAfter=14
    )
    style_h1 = ParagraphStyle(
        'H1', parent=styles['Normal'],
        fontName='Helvetica-Bold', fontSize=14, leading=17,
        textColor=C_PRIMARY, spaceBefore=14, spaceAfter=6, keepWithNext=True
    )
    style_h2 = ParagraphStyle(
        'H2', parent=styles['Normal'],
        fontName='Helvetica-Bold', fontSize=10.5, leading=13.5,
        textColor=C_ACCENT, spaceBefore=8, spaceAfter=4, keepWithNext=True
    )
    style_body = ParagraphStyle(
        'Body', parent=styles['Normal'],
        fontName='Helvetica', fontSize=8.5, leading=11.5,
        textColor=C_TEXT, spaceAfter=4
    )
    style_bullet = ParagraphStyle(
        'Bullet', parent=styles['Normal'],
        fontName='Helvetica', fontSize=8.5, leading=11.5,
        textColor=C_TEXT, leftIndent=12, firstLineIndent=-8, spaceAfter=2.5
    )
    style_bold_bullet = ParagraphStyle(
        'BoldBullet', parent=styles['Normal'],
        fontName='Helvetica-Bold', fontSize=8.5, leading=11.5,
        textColor=C_TEXT, leftIndent=12, firstLineIndent=-8, spaceAfter=2.5
    )
    style_qa_q = ParagraphStyle(
        'QAQuestion', parent=styles['Normal'],
        fontName='Helvetica-Bold', fontSize=8.8, leading=11.5,
        textColor=C_PRIMARY, spaceBefore=4, spaceAfter=1.5
    )
    style_qa_a = ParagraphStyle(
        'QAAnswer', parent=styles['Normal'],
        fontName='Helvetica', fontSize=8.5, leading=11.5,
        textColor=C_TEXT, leftIndent=10, spaceAfter=4
    )

    story = []

    # ---------------------------------------------------------
    # HEADER BANNER
    # ---------------------------------------------------------
    banner_data = [
        [Paragraph("<b>AGRISHIELD AI — MASTER PROJECT DEFENSE & PRESENTATION HANDBOOK</b>", ParagraphStyle('BText', fontName='Helvetica-Bold', fontSize=11, textColor=colors.white, alignment=1))],
        [Paragraph("A Comprehensive Technical & Viva Defense Guide for Project Team Members", ParagraphStyle('BSub', fontName='Helvetica', fontSize=8.5, textColor=colors.HexColor("#A7F3D0"), alignment=1))]
    ]
    banner_table = Table(banner_data, colWidths=[532])
    banner_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), C_PRIMARY),
        ('PADDING', (0,0), (-1,-1), 8),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('BOTTOMPADDING', (0,1), (-1,1), 8),
    ]))
    story.append(banner_table)
    story.append(Spacer(1, 10))

    # Meta Table: Team & Affiliations
    meta_data = [
        [
            Paragraph("<b>Project Title:</b> AI-Based Early Crop Stress Detection and Farmer Alert System", style_body),
            Paragraph("<b>Academic Year:</b> 2026 • Final Year Major Defense", style_body)
        ],
        [
            Paragraph("<b>Department:</b> Information Technology & Computer Engineering", style_body),
            Paragraph("<b>Agronomic Guidance:</b> Anand Agricultural University (AAU) & ICAR", style_body)
        ],
        [
            Paragraph("<b>Project Team:</b> Marmik Gandhi, Mehal Jain, Brij Patel, Aayush Barot", style_body),
            Paragraph("<b>Empirical Dataset:</b> 83,094 images across 4 benchmark datasets", style_body)
        ]
    ]
    meta_table = Table(meta_data, colWidths=[270, 262])
    meta_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), C_BG_CARD),
        ('BOX', (0,0), (-1,-1), 0.75, C_BORDER),
        ('INNERGRID', (0,0), (-1,-1), 0.5, C_BORDER),
        ('PADDING', (0,0), (-1,-1), 4.5),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 10))

    # ---------------------------------------------------------
    # SECTION 1: THE 30-SECOND ELEVATOR PITCH
    # ---------------------------------------------------------
    story.append(Paragraph("1. The 30-Second Elevator Pitch (What to Say First)", style_h1))
    pitch_text = (
        "<i>\"AgriShield AI is a full-stack, production-grade agricultural decision-support platform designed for Indian smallholder farmers. "
        "It takes a smartphone photograph of a crop leaf and performs three critical tasks in under 100 milliseconds: "
        "(1) Filters out unsuitable, dark, or non-plant images using an OpenCV Quality Guard; "
        "(2) Diagnoses foliar diseases using a Dual-CNN ensemble (EfficientNet-B0 + MobileNetV3) with <b>91.83% accuracy</b>; and "
        "(3) Pinpoints affected spots via a U-Net semantic segmentation network, calculates exact canopy damage percentage, "
        "explains its visual attention via Grad-CAM, and dispatches verified ICAR treatment prescriptions in English, Hindi, and Gujarati.\"</i>"
    )
    pitch_box = Table([[Paragraph(pitch_text, style_body)]], colWidths=[532])
    pitch_box.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), C_GREEN_BG),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#6EE7B7")),
        ('PADDING', (0,0), (-1,-1), 7),
    ]))
    story.append(pitch_box)
    story.append(Spacer(1, 8))

    # ---------------------------------------------------------
    # SECTION 2: PROBLEM STATEMENT & AGRICULTURAL CONTEXT
    # ---------------------------------------------------------
    story.append(Paragraph("2. Problem Statement & Real-World Motivation", style_h1))
    story.append(Paragraph("• <b>20% to 40% Annual Foliar Yield Losses:</b> Fungal and bacterial infections (early blight, leaf spot, rust) devastate hundreds of thousands of tons of crops across India every season, causing catastrophic financial losses to smallholder farmers.", style_bullet))
    story.append(Paragraph("• <b>Delayed Visual Detection:</b> Farmers typically recognize symptoms only when lesions cover >30% of the canopy. By then, containment requires heavy, expensive chemical fungicides.", style_bullet))
    story.append(Paragraph("• <b>Critical Expert Access Bottleneck:</b> The agricultural extension officer-to-farmer ratio in India exceeds 1:1,160. Physical laboratory diagnosis takes days, during which foliar blight can destroy an entire field.", style_bullet))
    story.append(Paragraph("• <b>Failure Modes of Existing AI Apps:</b> Standard mobile apps suffer from (1) Garbage-in garbage-out (accepting shoes or blurry images and predicting fake 99% diseases); (2) Black-box mistrust (no visual proof of why the model predicted a disease); and (3) Classification without localization (no measurement of what percentage of the leaf is damaged to calculate chemical dosage).", style_bullet))
    story.append(Spacer(1, 8))

    # ---------------------------------------------------------
    # SECTION 3: THE 8-STAGE END-TO-END PIPELINE
    # ---------------------------------------------------------
    story.append(Paragraph("3. The 8-Stage End-to-End Execution Pipeline", style_h1))
    pipeline_steps = [
        ("Stage 1 — Leaf Capture:", "Farmer captures or uploads a leaf photo via the mobile web browser interface."),
        ("Stage 2 — OpenCV Quality Guard:", "Pre-flight sanity filter: Laplacian variance (sharpness >= 65.0), mean brightness (35-230), and green/brown foliage ratio (>= 12% in HSV space). Rejects non-plant and blurry photos instantly."),
        ("Stage 3 — Dual-CNN Classification:", "EfficientNet-B0 (captures deep compound spatial features) and MobileNetV3-Small (fast edge features) independently evaluate the image across 38 crop-disease conditions."),
        ("Stage 4 — Soft-Voting Probability Fusion:", "Calibrated softmax probabilities are merged using soft-voting, boosting classification accuracy to 91.83%."),
        ("Stage 5 — U-Net Lesion Segmentation:", "A U-Net trained on the 7,774-pair PlantSeg benchmark isolates individual necrotic and chlorotic lesion boundaries and calculates pixel area (px²)."),
        ("Stage 6 — Grad-CAM Explainability:", "Computes gradient-weighted class activation maps from features[-1] of EfficientNet-B0 to visually verify that the model focused on real spots rather than background soil."),
        ("Stage 7 — Severity & Reliability Engine:", "Calculates canopy damage percentage (raw vs noise-filtered) and assigns a reliability badge (RELIABLE, APPROXIMATE, UNRELIABLE)."),
        ("Stage 8 — ICAR Advisory & Expert Triage:", "Prescribes chemical, biological, and cultural treatments. Low-confidence scans (<40%) or severe cases automatically route to the AAU Agronomist Triage Portal.")
    ]
    for s_title, s_desc in pipeline_steps:
        story.append(Paragraph(f"• <b>{s_title}</b> {s_desc}", style_bullet))
    story.append(Spacer(1, 8))

    # ---------------------------------------------------------
    # SECTION 4: RIGOROUS DATASET INVENTORY (ZERO LEAKAGE)
    # ---------------------------------------------------------
    story.append(Paragraph("4. Curated Dataset Foundation (83,094 Images) & Zero-Leakage Guarantee", style_h1))
    d_data = [
        ["Dataset Name", "Sample Count", "Nature of Data", "Core Purpose in Project"],
        ["1. PlantVillage", "54,305", "Controlled lab conditions, clean neutral backgrounds", "Baseline multi-crop disease classification (38 classes)"],
        ["2. PlantDoc", "2,573", "Real-world agricultural fields, natural weeds & soil", "Robustness to field background noise and daylight"],
        ["3. PlantSeg", "7,774", "Paired images with ground-truth binary masks", "Supervised training of U-Net lesion segmentation"],
        ["4. PlantWild", "18,542", "Uncurated in-the-wild agricultural imagery", "Generalization benchmark on complex foliage shadows"]
    ]
    d_table = Table([[Paragraph(f"<b>{c}</b>" if r==0 else c, style_body) for c in row] for r, row in enumerate(d_data)], colWidths=[85, 65, 182, 200])
    d_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), C_PRIMARY),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('BOX', (0,0), (-1,-1), 0.75, C_BORDER),
        ('INNERGRID', (0,0), (-1,-1), 0.5, C_BORDER),
        ('PADDING', (0,0), (-1,-1), 3.5),
    ]))
    story.append(d_table)
    story.append(Spacer(1, 4))
    story.append(Paragraph("<b>The Zero-Leakage Guarantee:</b> Many published papers report fake 99% accuracy because they randomly split PlantVillage without grouping photos of the same leaf. AgriShield AI used MD5 image hash grouping to force all identical or near-duplicate leaves strictly into the training split. The held-out test split is 100% untouched and unseen.", style_body))
    story.append(Spacer(1, 8))

    # ---------------------------------------------------------
    # SECTION 5: AI MODELS & MATHEMATICAL METHODOLOGY
    # ---------------------------------------------------------
    story.append(Paragraph("5. AI Architectures & Mathematical Foundations Explained Simply", style_h1))
    models_info = [
        ("1. EfficientNet-B0 (Compound Scaling):", "Instead of arbitrarily changing depth or width, EfficientNet scales depth, width, and resolution simultaneously using a compound coefficient (alpha * beta² * gamma² approx 2). It extracts rich spatial features and powers our Grad-CAM maps (83.68% standalone test accuracy)."),
        ("2. MobileNetV3-Small (Hard-Swish Activation):", "Engineered for low-latency CPU execution. Uses Inverted Residual blocks with linear bottlenecks and replaces expensive sigmoid functions with Hard-Swish: h-swish(x) = x * [ReLU6(x+3)/6], eliminating exponential computations (83.16% standalone test accuracy)."),
        ("3. Soft-Voting Probability Fusion:", "Combines the calibrated softmax probability vectors from both models: P_c = w1 * P_eff(c) + w2 * P_mob(c). Because the two models make uncorrelated errors on complex field leaves, probability fusion achieves an 8.15% accuracy jump to 91.83%."),
        ("4. U-Net Lesion Segmentation:", "Encoder contracting path captures deep contextual features; expanding decoder restores full spatial resolution. Skip connections pass high-resolution feature maps across layers to preserve fine, irregular lesion edges. Trained with combined BCE + Dice Loss."),
        ("5. Grad-CAM (Visual Explainability):", "Computes the gradient of the predicted class score with respect to the last convolutional layer (features[-1]). Displays a color-coded heatmap confirming that the model looked at actual foliar lesions rather than background soil.")
    ]
    for m_title, m_desc in models_info:
        story.append(Paragraph(f"• <b>{m_title}</b> {m_desc}", style_bullet))
    story.append(Spacer(1, 8))

    # ---------------------------------------------------------
    # SECTION 6: EMPIRICAL BENCHMARK METRICS
    # ---------------------------------------------------------
    story.append(Paragraph("6. Empirical Benchmark Results (Numbers to Defend)", style_h1))
    res_data = [
        ["Model Architecture", "Test Accuracy", "Macro Precision", "Macro Recall", "Macro F1 Score"],
        ["EfficientNet-B0", "83.68%", "0.817", "0.816", "0.812"],
        ["MobileNetV3-Small", "83.16%", "0.814", "0.810", "0.806"],
        ["Dual-CNN Soft Ensemble", "91.83%", "0.894", "0.887", "0.889"],
        ["U-Net Segmentation", "31.42% Mean Dice", "20.66% Mean IoU", "Tested on Non-Empty Masks", "PlantSeg Benchmark"]
    ]
    res_table = Table([[Paragraph(f"<b>{c}</b>" if r==0 else c, style_body) for c in row] for r, row in enumerate(res_data)], colWidths=[130, 85, 105, 105, 107])
    res_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), C_PRIMARY),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('BACKGROUND', (0,3), (-1,3), C_GREEN_BG),
        ('BOX', (0,0), (-1,-1), 0.75, C_BORDER),
        ('INNERGRID', (0,0), (-1,-1), 0.5, C_BORDER),
        ('PADDING', (0,0), (-1,-1), 3.5),
    ]))
    story.append(res_table)
    story.append(Spacer(1, 4))
    story.append(Paragraph("<b>Why We Reject 98%-99% Claims:</b> Many papers inflate accuracy by testing on duplicate leaves, or inflate Dice scores by testing on healthy leaves with empty masks (predicting 0 spots on an empty leaf scores 100%). Our 91.83% accuracy and 31.42% Dice score represent true, unpadded performance on isolated, non-overlapping field data.", style_body))
    story.append(Spacer(1, 8))

    # ---------------------------------------------------------
    # SECTION 7: FULL-STACK SOFTWARE ARCHITECTURE
    # ---------------------------------------------------------
    story.append(Paragraph("7. Full-Stack Software Engineering Architecture", style_h1))
    story.append(Paragraph("• <b>Backend Framework:</b> Python 3.12 with FastAPI and Uvicorn ASGI server for asynchronous high-throughput execution.", style_bullet))
    story.append(Paragraph("• <b>Database Tier:</b> SQLAlchemy 2.0 ORM with 11 relational tables (users, scans, detections, bounding_boxes, alerts, triage_reviews, etc.). Runs on SQLite locally and switches seamlessly to PostgreSQL 15 via Docker Compose.", style_bullet))
    story.append(Paragraph("• <b>Security:</b> Direct native Bcrypt password hashing (bcrypt.checkpw) and JWT Bearer token authentication.", style_bullet))
    story.append(Paragraph("• <b>Frontend Tier:</b> React 19, TypeScript, Vite, and Tailwind CSS. Clean SaaS agricultural theme (emerald green & neutral white; no heavy glassmorphism or fake charts).", style_bullet))
    story.append(Paragraph("• <b>Localization:</b> Complete trilingual support across English, Hindi (हिन्दी), and Gujarati (ગુજરાતી).", style_bullet))
    story.append(Paragraph("• <b>Live Integrations:</b> Agmarknet APMC Mandi commodity rates + OpenWeather live disease vulnerability forecasting (alerts when humidity > 85%).", style_bullet))
    story.append(Spacer(1, 8))

    # ---------------------------------------------------------
    # SECTION 8: TEAM MEMBER ROLE ALLOCATION
    # ---------------------------------------------------------
    story.append(Paragraph("8. Team Member Slide Allocation & Speaking Roles", style_h1))
    roles_data = [
        ["Member Name", "Engineering Role", "Slides", "Core Topics to Explain"],
        ["Marmik Gandhi", "Team Leader • Full-Stack", "1, 2, 3, 4", "Agricultural problem, 20-40% losses, project objectives, 4-tier architecture."],
        ["Brij Patel", "Backend & Computer Vision", "5, 7", "Technology stack, 4 datasets, zero leakage, OpenCV Quality Guard & Grad-CAM evidence."],
        ["Mehal Jain", "AI / Machine Learning", "6, 8", "Math of EfficientNet, MobileNet, U-Net, soft-voting fusion, empirical 91.83% results."],
        ["Aayush Barot", "Frontend & UX/Localization", "9, 10, 11", "5-stage workflow, Kisan Bot, Mandi/Weather, Agronomist Portal, limitations, Q&A."]
    ]
    roles_table = Table([[Paragraph(f"<b>{c}</b>" if r==0 else c, style_body) for c in row] for r, row in enumerate(roles_data)], colWidths=[95, 115, 55, 267])
    roles_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), C_PRIMARY),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('BOX', (0,0), (-1,-1), 0.75, C_BORDER),
        ('INNERGRID', (0,0), (-1,-1), 0.5, C_BORDER),
        ('PADDING', (0,0), (-1,-1), 3.5),
    ]))
    story.append(roles_table)
    story.append(Spacer(1, 8))

    # ---------------------------------------------------------
    # SECTION 9: TOP 12 EXAMINER / JURY QUESTIONS & KILLER ANSWERS
    # ---------------------------------------------------------
    story.append(Paragraph("9. Top 12 Tough Examiner / Jury Questions & Killer Answers", style_h1))
    qas = [
        ("Q1: Why use an ensemble of EfficientNet and MobileNet instead of a single larger model like ResNet-101 or ViT?",
         "Large models like ResNet-101 or Vision Transformers have over 40 million parameters, causing severe overfitting on plant pathology datasets and unacceptable CPU latency. EfficientNet-B0 captures deep spatial features, while MobileNetV3 captures rapid edge features. Because their errors are uncorrelated, soft-voting probability fusion boosts accuracy from ~83% to 91.83% while keeping CPU inference under 100 milliseconds."),
        ("Q2: Why is the U-Net Dice score 31.42% and not 80% or 90%?",
         "Most literature inflates Dice scores by evaluating on healthy leaves with empty masks (predicting zero spots on an empty leaf scores 100%). Our 31.42% Dice and 20.66% IoU were evaluated strictly on non-empty, genuine lesion masks from the challenging PlantSeg benchmark. In agricultural pathology, tiny, irregular lesion boundaries under variable lighting make 31.4% true Dice a solid, honest baseline for canopy percentage estimation."),
        ("Q3: What prevents a farmer from uploading a picture of a dog, shoe, or blurry photo?",
         "Our OpenCV Quality Guard filters images before they reach the neural networks. It computes Laplacian variance to reject blurry images (<65.0), tests mean grayscale intensity to reject underexposed/overexposed lighting (<35 or >230), and calculates an HSV color mask to verify at least 12% foliar vegetation. If any check fails, the photo is immediately rejected with clear corrective guidance."),
        ("Q4: How did you prevent data leakage during model training?",
         "Photographers often take several photos of the same diseased leaf from different angles. Random splitting puts identical leaves into both training and testing sets, artificially inflating accuracy to 99%. We computed MD5 hash signatures to ensure all near-duplicates were strictly confined to the training split, keeping the held-out test split completely untouched."),
        ("Q5: What is Grad-CAM, and why is it necessary?",
         "Grad-CAM stands for Gradient-weighted Class Activation Mapping. It tracks gradients flowing into the final convolutional layer (features[-1]) to highlight the exact pixels that triggered the classification. It gives agronomists and farmers visual proof that the model looked at real foliar lesions rather than background soil or shadows."),
        ("Q6: How is disease severity calculated?",
         "Our severity engine combines U-Net lesion pixel counts with the OpenCV leaf mask: Affected Canopy % = (Diseased Lesion Pixels / Total Leaf Pixels) * 100. We also extract individual contour bounding boxes to count discrete lesion clusters and report both raw and noise-filtered percentages without erasing legitimate symptoms."),
        ("Q7: What happens if the AI is uncertain about a leaf?",
         "If the ensemble confidence is below 40%, the system flags the scan as UNCERTAIN, displays a prominent advisory warning to the farmer, and automatically routes the scan and telemetry to our AAU Agronomist Triage Portal, where a human expert can review the case and issue an official prescription."),
        ("Q8: What database are you using, and how does it scale?",
         "We use SQLAlchemy 2.0 ORM with 11 relational tables. In development, it runs on SQLite. In production, it connects seamlessly to PostgreSQL 15 via Docker Compose with zero code changes."),
        ("Q9: How does the Kisan AI Bot provide reliable advice without hallucinating?",
         "The Kisan AI Bot is grounded in official ICAR and Anand Agricultural University (AAU) plant pathology guidelines. Every response cites the specific handbook reference, ensuring farmers receive verified chemical dilutions and organic biocontrol methods."),
        ("Q10: How are weather and mandi prices integrated?",
         "We connect to the OpenWeather API to monitor relative humidity, temperature, and rainfall in Anand District. When humidity exceeds 85% with warm temperatures, our epidemiological model alerts farmers to imminent fungal blight risk. Simultaneously, we pull live APMC commodity market rates via Agmarknet to assist farmers with harvest decisions."),
        ("Q11: What are the current limitations of your system?",
         "The three main limitations are: (1) Severe leaf overlap in dense canopies can partially obscure lesion boundaries; (2) Sub-millimeter fungal pinheads are hard to distinguish from soil dust on low-cost smartphone sensors; and (3) Harsh high-noon sunlight creates high-contrast shadows that can occasionally trigger false edge contours."),
        ("Q12: What is your future scope?",
         "We plan to: (1) Quantize our models using TFLite and ONNX for full offline execution on low-cost Android phones; (2) Integrate drone-mounted multispectral NIR cameras for acre-scale stress mapping; and (3) Fuse visual foliar data with real-time IoT soil moisture and NPK telemetry.")
    ]
    for q, a in qas:
        story.append(Paragraph(f"• <b>{q}</b>", style_qa_q))
        story.append(Paragraph(f"<b>Answer:</b> <i>\"{a}\"</i>", style_qa_a))

    # Build document
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Handbook PDF generated successfully at: {OUTPUT_PDF}")

if __name__ == "__main__":
    build_pdf()
