from rest_framework import serializers

class PredictionSerializer(serializers.Serializer):
    """
    Serializer for validating input data for predictions.
    """
    abstract = serializers.CharField(
        max_length=5000,
        required=True,
        allow_blank=False,
        trim_whitespace=True,
        help_text="Abstract text to be analyzed by the model. Maximum length: 5000 characters."
    )

    def validate_abstract(self, value):
        """
        Custom validation for the 'abstract' field.

        Ensures the input text meets certain criteria such as:
        - Minimum length requirement.
        - Non-empty content after trimming.
        - Absence of potentially unsupported characters.

        Args:
            value (str): The input text provided in the request.

        Returns:
            str: The validated and cleaned text.

        Raises:
            serializers.ValidationError: If the input text is invalid.
        """
        # Check for minimum length
        min_length = 50  # Example minimum length
        if len(value.strip()) < min_length:
            raise serializers.ValidationError(
                f"The abstract is too short. It must be at least {min_length} characters long."
            )

        # Check for unsupported characters (example: emojis)
        if any(ord(char) > 127 for char in value):
            raise serializers.ValidationError(
                "The abstract contains unsupported characters. Only ASCII text is allowed."
            )

        # Additional checks can go here (e.g., profanity filters, banned words, etc.)
        return value

