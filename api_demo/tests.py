from django.test import TestCase
from rest_framework.test import APIClient
from rest_framework import status
from api_demo.serializers import PredictionSerializer

# Unit tests
class PredictionSerializerTestCase(TestCase):
    def test_serializer_valid_input(self):
        """
        Test serializer with valid input data.
        """
        valid_data = {"abstract": "A detailed and valid scientific abstract that exceeds 50 characters."}
        serializer = PredictionSerializer(data=valid_data)
        self.assertTrue(serializer.is_valid(), "Serializer should accept valid input.")

    def test_serializer_invalid_input(self):
        """
        Test serializer with invalid input (too short).
        """
        invalid_data = {"abstract": "Short abstract"}
        serializer = PredictionSerializer(data=invalid_data)
        self.assertFalse(serializer.is_valid(), "Serializer should reject input shorter than 50 characters.")
        self.assertIn("abstract", serializer.errors, "Errors should contain 'abstract' key.")
        self.assertIn("too short", str(serializer.errors["abstract"]), "Error should mention 'too short'.")


class PredictViewTestCase(TestCase):
    def setUp(self):
        """
        Set up an API client for testing.
        """
        self.client = APIClient()
        self.url = "/api/predict/"  # Update this to match your endpoint

    def test_predict_view_valid_request(self):
        """
        Test PredictView with valid input data.
        """
        valid_data = {"abstract": "A valid scientific abstract that exceeds the minimum character limit."}
        response = self.client.post(self.url, data=valid_data, format="json")
        self.assertEqual(response.status_code, status.HTTP_200_OK, "View should return 200 for valid input.")
        self.assertIn("predictions", response.data, "Response should include 'predictions' key.")

    def test_predict_view_invalid_request(self):
        """
        Test PredictView with invalid input data.
        """
        invalid_data = {"abstract": "Too short"}
        response = self.client.post(self.url, data=invalid_data, format="json")
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST, "View should return 400 for invalid input.")
        self.assertIn("abstract", response.data, "Errors should include 'abstract' key.")

