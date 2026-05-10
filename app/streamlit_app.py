import requests
import streamlit as st

from PIL import Image
from io import BytesIO


API_URL = "http://127.0.0.1:8000"


st.set_page_config(
    page_title="Smart Parking Vision",
    page_icon="🅿",
    layout="wide",
    initial_sidebar_state="expanded"
)


st.markdown(
    """
    <style>

    .stApp {
        background-color: #0f172a;
        color: white;
    }

    .main-title {
        font-size: 52px;
        font-weight: 900;
        color: white;
        margin-bottom: 0px;
    }

    .subtitle {
        font-size: 20px;
        color: #94a3b8;
        margin-bottom: 30px;
    }

    .metric-card {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 18px;
        padding: 20px;
        text-align: center;
    }

    .metric-value {
        font-size: 42px;
        font-weight: 800;
        color: white;
    }

    .metric-label {
        color: #94a3b8;
        font-size: 15px;
    }

    .status-ok {
        color: #22c55e;
        font-weight: 700;
        font-size: 18px;
    }

    .status-error {
        color: #ef4444;
        font-weight: 700;
        font-size: 18px;
    }

    .section-title {
        font-size: 28px;
        font-weight: 800;
        margin-bottom: 15px;
        color: white;
    }

    .stButton > button {
        background-color: #2563eb !important;
        color: white !important;
        border: none !important;
        border-radius: 14px !important;
        font-size: 17px !important;
        font-weight: 700 !important;
        padding: 14px 20px !important;
        transition: 0.2s ease-in-out !important;
    }

    .stButton > button:hover {
        background-color: #1d4ed8 !important;
        color: white !important;
        border: none !important;
        transform: translateY(-1px);
    }

    .stButton > button:active {
        background-color: #1e40af !important;
        color: white !important;
    }

    </style>
    """,
    unsafe_allow_html=True
)


def check_api_status():
    try:
        response = requests.get(f"{API_URL}/health", timeout=3)
        return response.status_code == 200
    except Exception:
        return False


def analyze_image(uploaded_file):
    files = {
        "file": (
            uploaded_file.name,
            uploaded_file.getvalue(),
            uploaded_file.type
        )
    }

    response = requests.post(
        f"{API_URL}/analyze-parking",
        files=files,
        timeout=120
    )

    response.raise_for_status()
    return response.json()


def load_result_image(result_image_path):
    image_url = f"{API_URL}{result_image_path}"

    response = requests.get(
        image_url,
        timeout=30
    )

    response.raise_for_status()

    return Image.open(
        BytesIO(response.content)
    )


with st.sidebar:
    st.title("🚗 Smart Parking")

    st.markdown("---")

    api_ok = check_api_status()

    if api_ok:
        st.markdown(
            '<p class="status-ok">Backend API online</p>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<p class="status-error">Backend API offline</p>',
            unsafe_allow_html=True
        )

    st.markdown("---")

    st.subheader("System")

    st.write("• FastAPI backend")
    st.write("• Streamlit frontend")
    st.write("• OpenCV visualization")
    st.write("• YOLO-ready pipeline")
    st.write("• Live video analytics")

    st.markdown("---")

    st.subheader("Project")

    st.write("AI parking occupancy detection")
    st.write("Smart campus / mall / office parking analytics")


st.markdown(
    '<div class="main-title">Smart Parking Vision</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">AI-powered parking occupancy monitoring system</div>',
    unsafe_allow_html=True
)


left_col, right_col = st.columns([1, 1.6])


with left_col:
    st.markdown(
        '<div class="section-title">Upload image</div>',
        unsafe_allow_html=True
    )

    uploaded_file = st.file_uploader(
        "Upload parking image",
        type=["jpg", "jpeg", "png"]
    )

    analyze_button = st.button(
        "Analyze parking",
        use_container_width=True,
        type="primary"
    )


with right_col:
    st.markdown(
        '<div class="section-title">Preview</div>',
        unsafe_allow_html=True
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file)

        st.image(
            image,
            use_container_width=True
        )
    else:
        st.info("Upload image to preview.")


if uploaded_file is not None and analyze_button:
    with st.spinner("AI system analyzing parking..."):
        try:
            result = analyze_image(uploaded_file)
            statistics = result["statistics"]

            st.markdown("---")

            st.markdown(
                '<div class="section-title">Analytics</div>',
                unsafe_allow_html=True
            )

            metric1, metric2, metric3, metric4 = st.columns(4)

            with metric1:
                st.markdown(
                    f'''
                    <div class="metric-card">
                        <div class="metric-value">{statistics["total_spots"]}</div>
                        <div class="metric-label">Total spots</div>
                    </div>
                    ''',
                    unsafe_allow_html=True
                )

            with metric2:
                st.markdown(
                    f'''
                    <div class="metric-card">
                        <div class="metric-value">{statistics["free_spots"]}</div>
                        <div class="metric-label">Free spots</div>
                    </div>
                    ''',
                    unsafe_allow_html=True
                )

            with metric3:
                st.markdown(
                    f'''
                    <div class="metric-card">
                        <div class="metric-value">{statistics["occupied_spots"]}</div>
                        <div class="metric-label">Occupied spots</div>
                    </div>
                    ''',
                    unsafe_allow_html=True
                )

            occupancy_percent = statistics["occupancy_rate"] * 100

            with metric4:
                st.markdown(
                    f'''
                    <div class="metric-card">
                        <div class="metric-value">{occupancy_percent:.1f}%</div>
                        <div class="metric-label">Occupancy</div>
                    </div>
                    ''',
                    unsafe_allow_html=True
                )

            st.markdown("### Parking occupancy")

            st.progress(
                min(
                    max(
                        statistics["occupancy_rate"],
                        0.0
                    ),
                    1.0
                )
            )

            result_image = load_result_image(
                result["result_image"]
            )

            st.markdown("---")

            st.markdown(
                '<div class="section-title">AI visualization</div>',
                unsafe_allow_html=True
            )

            st.subheader("AI detection result")

            st.image(
                result_image,
                use_container_width=True
            )

            st.success("Parking analysis completed successfully")

        except Exception as e:
            st.error(str(e))


st.markdown("---")

st.markdown(
    '<div class="section-title">Live video analytics</div>',
    unsafe_allow_html=True
)

video_file = st.file_uploader(
    "Upload parking video",
    type=["mp4", "avi", "mov"],
    key="video_uploader"
)

if video_file is not None:
    st.video(video_file)

    if st.button(
        "Analyze video",
        use_container_width=True,
        type="primary"
    ):
        with st.spinner("Analyzing video stream..."):
            try:
                files = {
                    "file": (
                        video_file.name,
                        video_file.getvalue(),
                        video_file.type
                    )
                }

                response = requests.post(
                    f"{API_URL}/analyze-video",
                    files=files,
                    timeout=300
                )

                response.raise_for_status()
                result = response.json()

                st.success("Video processed successfully")

                st.subheader("Video statistics")

                metric_col1, metric_col2 = st.columns(2)

                with metric_col1:
                    st.metric(
                        "Processed frames",
                        result["processed_frames"]
                    )

                with metric_col2:
                    st.metric(
                        "Timeline points",
                        len(result["free_history"])
                    )

                st.subheader("Parking occupancy timeline")

                chart_data = {
                    "Free": result["free_history"],
                    "Occupied": result["occupied_history"]
                }

                st.line_chart(chart_data)

            except Exception as e:
                st.error(str(e))