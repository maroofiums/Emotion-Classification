import streamlit as st

from src.inference import EmotionPredictor


# Page Configuration
st.set_page_config(
    page_title="Emotion Classifier",
    page_icon="",
    layout="centered",
)


# Load Model
@st.cache_resource
def load_predictor():
    """
    Load the model once and cache it.
    """

    return EmotionPredictor()


# Application
def main():

    st.title("Emotion Classification")

    st.write(
        "A Bidirectional LSTM model that classifies "
        "text into six emotions."
    )

    st.divider()

    # Model
    try:
        predictor = load_predictor()

    except FileNotFoundError as error:

        st.error(str(error))

        st.stop()

    # Text Input
    text = st.text_area(
        "Enter your text",
        placeholder=(
            "Example: "
            "I finally achieved my goal!"
        ),
        height=150,
    )

    # Prediction
    if st.button(
        "Predict Emotion",
        type="primary",
        use_container_width=True,
    ):

        if not text.strip():

            st.warning(
                "Please enter some text."
            )

            st.stop()

        try:

            result = predictor.predict(
                text
            )

        except Exception as error:

            st.error(
                f"Prediction failed: {error}"
            )

            st.stop()

            # Result
        emotion = result[
            "emotion"
        ]

        confidence = result[
            "confidence"
        ]

        probabilities = result[
            "probabilities"
        ]

        st.divider()

        st.subheader(
            "Prediction"
        )

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                label="Emotion",
                value=emotion.title(),
            )

        with col2:

            st.metric(
                label="Confidence",
                value=f"{confidence:.2%}",
            )

            # Probability Chart
        st.subheader(
            "Emotion Probabilities"
        )

        chart_data = {
            emotion_name.title(): probability
            for emotion_name, probability
            in probabilities.items()
        }

        st.bar_chart(
            chart_data,
            horizontal=True,
        )

            # Detailed Probabilities
        st.subheader(
            "Detailed Probabilities"
        )

        for (
            emotion_name,
            probability,
        ) in probabilities.items():

            st.write(
                f"**{emotion_name.title()}** "
                f"— {probability:.2%}"
            )

            st.progress(
                probability
            )

    # About
    st.divider()

    with st.expander(
        "About this model"
    ):

        st.write(
            """
            This application uses a Bidirectional LSTM
            neural network for emotion classification.

            The model predicts six emotions:

            - Sadness
            - Joy
            - Love
            - Anger
            - Fear
            - Surprise

            The model was trained using the
            dair-ai/emotion dataset.
            """
        )


if __name__ == "__main__":
    main()