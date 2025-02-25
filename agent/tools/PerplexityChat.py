import requests
import json


class PerplexityChat:
    BASE_URL = "https://api.perplexity.ai/chat/completions"

    def __init__(self, pplx_model="sonar", pplx_api_key=None):
        self.api_key = pplx_api_key
        self.model = pplx_model

    def invoke(self, system_prompt, query, max_tokens=1000):
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
