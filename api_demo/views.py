from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from api_demo.serializers import PredictionSerializer
from api_demo.helpers import ModelLoader
from api_demo.preprocessors import preprocess_text
import numpy as np
import logging

# Configure logging
logger = logging.getLogger(__name__)

# Load the model onlt once, not on every request
model = ModelLoader.get_model()
logger.info("Model loaded successfully.")

class PredictView(APIView):
    """
    Handles POST requests for predictions.
    """

    def post(self, request, *args, **kwargs):
        """
        Handles the POST request to predict using the model.
        """
        serializer = PredictionSerializer(data=request.data)

        if not serializer.is_valid():
            logger.error(f"Invalid input: {serializer.errors}")
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        try:
            # Preprocess input
            abstract = serializer.validated_data["abstract"]
            input_ids, attention_mask = preprocess_text(abstract)
            logger.debug(f"Preprocessed input: input_ids={input_ids.shape}, attention_mask={attention_mask.shape}")

            # Perform prediction
            predictions = model.predict({"input_ids": input_ids, "attention_mask": attention_mask})
            logger.debug(f"Raw predictions: {predictions}")

            # Post-process predictions
            predicted_classes = self.postprocess(predictions)
            return Response({"predictions": predicted_classes}, status=status.HTTP_200_OK)

        except ValueError as ve:
            logger.error(f"ValueError: {ve}")
            return Response({"error": str(ve)}, status=status.HTTP_400_BAD_REQUEST)
        except Exception as e:
            logger.exception("Unexpected error during prediction.")
            return Response({"error": "An error occurred while processing the request."}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    @staticmethod
    def postprocess(predictions):
        """
        Post-processes raw model predictions into a more usable format.

        Args:
            predictions (numpy.ndarray): Raw model output.

        Returns:
            list: List of predicted classes or scores.
        """
        # Example: Convert predictions to a Python list
        return predictions.tolist()

