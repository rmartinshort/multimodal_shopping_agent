import requests
import json


class PerplexityChat:
    """
    A client for interacting with the Perplexity AI chat completions API.

    This class provides methods to initialize a connection to the Perplexity API
    and make requests to generate AI-powered chat completions.

    Attributes:
        BASE_URL (str): The endpoint URL for the Perplexity chat completions API.
        api_key (str): The API key used for authentication with Perplexity.
        model (str): The Perplexity model to use for completions.
    """

    BASE_URL = "https://api.perplexity.ai/chat/completions"

    def __init__(self, pplx_model="sonar", pplx_api_key=None):
        self.api_key = pplx_api_key
        self.model = pplx_model

    def invoke(self, system_prompt, query, max_tokens=1000):
        """
        Send a request to the Perplexity API to generate a chat completion.

        Args:
            system_prompt (object): An object containing the system_prompt attribute
                that defines the behavior of the AI assistant.
            query (str): The user's input message or query.
            max_tokens (int, optional): Maximum number of tokens to generate. Defaults to 1000.

        Returns:
            requests.Response: The HTTP response from the Perplexity API.
        """
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt.system_prompt},
                {"role": "user", "content": query},
            ],
            "max_tokens": max_tokens,
            "temperature": 0,
            "top_p": 0.9,
            "search_domain_filter": None,
            "return_images": False,
            "return_related_questions": False,
            "stream": False,
            "presence_penalty": 0,
            "frequency_penalty": 1,
            "response_format": None,
        }
        headers = {
            "Authorization": "Bearer {}".format(self.api_key),
            "Content-Type": "application/json",
        }

        response = requests.request(
            "POST", self.BASE_URL, json=payload, headers=headers
        )

        return response

    @staticmethod
    def craft_text_response(response):
        """
        Process the API response and extract the generated text and citations.

        Args:
            response (requests.Response): The HTTP response from the Perplexity API.

        Returns:
            str: The formatted text response with citations, or an error message
                if the request was unsuccessful.
        """
        if response.status_code == 200:
            response_json = json.loads(response.text)
            citations = str(
                {i + 1: x for i, x in enumerate(response_json["citations"])}
            )
            text_response = response_json["choices"][0]["message"]["content"]
            result = text_response + f"\ncitations: \n{citations}"

        else:
            result = "Unable to call web search, got an error"

        return result
