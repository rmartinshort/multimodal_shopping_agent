from typing import Annotated
import logging
import datetime
from PIL import ImageGrab
from typing import List
from livekit.agents import llm
import webbrowser
from image_utils import encode_image, capture_image_from_video_stream
from config import IMAGE_MODEL, IMAGE_RESIZE_WIDTH
from prompts import WebSearchLLMPrompt, ScreenshotImagePrompt, VideoStreamImagePrompt

logger = logging.getLogger("agent_tools")
logger.setLevel(logging.INFO)


class AgentTools(llm.FunctionContext):
    def __init__(
        self, room, image_model, web_model, database, user_id, conversation_id
    ):
        super().__init__()
        self._room = room
        self._image_llm = image_model
        self._web_model = web_model
        self._user_id = user_id
        self._conversation_id = conversation_id
        self.db = database

    @property
    def room(self):
        return self._room

    @property
    def image_llm_client(self):
        return self._image_llm

    @property
    def web_model(self):
        return self._web_model

    @property
    def user_id(self):
        return self._user_id

    @property
    def conversation_id(self):
        return self._conversation_id

    @llm.ai_callable()
    async def get_todays_date_and_time(self):
        """
        When we want to get the date or time
        """
        logger.info("Getting today's date and time")
        date_time = str(datetime.datetime.now())
        return f"The time is {date_time}"

    @llm.ai_callable()
    async def open_urls(
        self,
        urls_to_open: Annotated[
            List[str],
            llm.TypeInfo(
                description="The list of urls that we want to open. Each one will be opened in a new browser tab"
            ),
        ],
    ):
        """
        Use when we want to navigate to some urls. A typical use case will be after you search online for something. That query
        should return some citations, which are urls that you can copy past into the urls_to_open list. You then call this tool
        to open the urls so that the user can see them
        """
        logger.info(
            f"CALL OPEN URLS: Here are the sites to navigate to: {urls_to_open}"
        )
        for url in urls_to_open:
            webbrowser.open(url, new=2, autoraise=True)

        self.db.store_text(
            user_id=self.user_id,
            conversation_id=self.conversation_id,
            tool_id="open_urls",
            data_type="urls_list",
            text_data=str(urls_to_open),
        )

        return "Done opening urls"

    @llm.ai_callable()
    async def search_the_web(
        self,
        user_question: Annotated[
            str,
            llm.TypeInfo(
                description="The question we want to ask or subject we want to learn about, using a call to a model that has access to the internet"
            ),
        ],
    ):
        """
        Use when you have a question or topic where up-to-date information is required. This will call an LLM that has
        access to the internet and return its answer to your question in addition to the sources it used
        """
        logger.info(f"CALL PPLX: Here's whats going to be asked: {user_question}")
        result = self.web_model.invoke(
            query=user_question, system_prompt=WebSearchLLMPrompt
        )
        text_result = self.web_model.craft_text_response(result)

        logger.info(f"CALL PPLX: Here's the response {text_result}")
        self.db.store_text(
            user_id=self.user_id,
            conversation_id=self.conversation_id,
            tool_id="search_the_web",
            data_type="input",
            text_data=user_question,
        )
        self.db.store_text(
            user_id=self.user_id,
            conversation_id=self.conversation_id,
            tool_id="search_the_web",
            data_type="output",
            text_data=text_result,
        )
        # save raw model result
        self.db.store_text(
            user_id=self.user_id,
            conversation_id=self.conversation_id,
            tool_id="search_the_web",
            data_type="metadata",
            text_data=str(result),
        )
        return text_result

    @llm.ai_callable()
    async def question_camera_image(
        self,
        user_question: Annotated[
            str,
            llm.TypeInfo(
                description="The question we want to ask about a frame captured from the user's video stream"
            ),
        ],
    ):
        """
        Grab a frame from the user's video stream and use an image model to ask a question of it. The use case for this will be if the user
        has an item they want to learn more about.
        """

        logger.info(f"CAPTURE FRAME: Here's whats going to be asked: {user_question}")
        latest_frame = await capture_image_from_video_stream(self._room)

        # if the image is present, then ask a question of it
        if not isinstance(latest_frame["b64_image"], type(None)):
            base64_image = latest_frame["b64_image"]

            try:
                response = self._image_llm.chat.completions.create(
                    model=IMAGE_MODEL,
                    messages=[
                        {
                            "role": "system",
                            "content": [
                                {
                                    "type": "text",
                                    "text": VideoStreamImagePrompt.system_message,
                                }
                            ],
                        },
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": f"{user_question}"},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{base64_image}"
                                    },
                                },
                            ],
                        },
                    ],
                    max_tokens=1000,
                )
                final_response_text = response.choices[0].message.content
                token_usage = str(response.usage.__dict__)
                logger.info(f"CAPTURE FRAME: Here's the response {final_response_text}")
            except Exception as e:
                logger.info(e)
                token_usage = None
                final_response_text = "Unable to ask a question of this captured video frame: Technical difficulties"
                base64_image = None

        else:
            token_usage = None
            final_response_text = "Unable to ask question about frame because byte64 encoded image not present"
            base64_image = None

        self.db.store_text(
            user_id=self.user_id,
            conversation_id=self.conversation_id,
            tool_id="question_camera_image",
            data_type="input",
            text_data=user_question,
        )
        self.db.store_image(
            user_id=self.user_id,
            conversation_id=self.conversation_id,
            tool_id="question_camera_image",
            image_data=base64_image,
        )
        self.db.store_text(
            user_id=self.user_id,
            conversation_id=self.conversation_id,
            tool_id="question_camera_image",
            data_type="output",
            text_data=final_response_text,
        )
        # save raw model result
        self.db.store_text(
            user_id=self.user_id,
            conversation_id=self.conversation_id,
            tool_id="question_camera_image",
            data_type="metadata",
            text_data=token_usage,
        )

        return final_response_text

    @llm.ai_callable()
    async def question_screenshot(
        self,
        user_question: Annotated[
            str,
            llm.TypeInfo(
                description="The question we want to ask about the screenshot"
            ),
        ],
    ):
        """
        Take a screenshot and then use an image model to ask a question of it. Use this tool when the user asks you to help them with a task
        that's on their screen. They might be asking if the price of some product seems reasonable, whether they can get a better deal elsewhere
        or something of the like.
        """

        screenshot = ImageGrab.grab().convert("RGB")
        width, height = screenshot.size
        new_height = int(height * IMAGE_RESIZE_WIDTH / width)
        screenshot = screenshot.resize((IMAGE_RESIZE_WIDTH, new_height))
        base64_image = encode_image(screenshot)
        logger.info(f"SCREENSHOT: Here's whats going to be asked {user_question}")

        try:
            response = self._image_llm.chat.completions.create(
                model=IMAGE_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": [
                            {
                                "type": "text",
                                "text": ScreenshotImagePrompt.system_message,
                            }
                        ],
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": f"{user_question}"},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                },
                            },
                        ],
                    },
                ],
                max_tokens=1000,
            )
            final_response_text = response.choices[0].message.content
            token_usage = str(response.usage.__dict__)
            logger.info(f"SCREENSHOT: Here's the response {final_response_text}")
        except Exception as e:
            logger.info(e)
            token_usage = None
            final_response_text = "Unable to ask a question of this captured screenshot: Technical difficulties"
            base64_image = None

        self.db.store_text(
            user_id=self.user_id,
            conversation_id=self.conversation_id,
            tool_id="question_screenshot",
            data_type="input",
            text_data=user_question,
        )
        self.db.store_image(
            user_id=self.user_id,
            conversation_id=self.conversation_id,
            tool_id="question_screenshot",
            image_data=base64_image,
        )
        self.db.store_text(
            user_id=self.user_id,
            conversation_id=self.conversation_id,
            tool_id="question_screenshot",
            data_type="output",
            text_data=final_response_text,
        )
        # save raw model result
        self.db.store_text(
            user_id=self.user_id,
            conversation_id=self.conversation_id,
            tool_id="question_screenshot",
            data_type="metadata",
            text_data=token_usage,
        )

        return final_response_text
