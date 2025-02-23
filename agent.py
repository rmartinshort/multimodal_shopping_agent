from __future__ import annotations

import logging
from dotenv import load_dotenv

from livekit import rtc
from livekit.agents import (
    AutoSubscribe,
    JobContext,
    WorkerOptions,
    cli,
    llm,
)
from livekit.agents.multimodal import MultimodalAgent
from livekit.plugins import openai
from openai import OpenAI
import os
from PerplexityChat import PerplexityChat
from AgentTools import AgentTools
from prompts import RealTimeModelDriverPrompt
from AgentDatabase import AgentDatabase
from ConversationLogger import ConversationLogger
from config import (
    REALTIME_MODEL,
    REALTIME_TEMPERATURE,
    VOICE,
    PPLX_MODEL,
    TOOL_DATABASE_NAME,
    CONVERSATION_LOG_PREFIX,
)
import uuid

load_dotenv(dotenv_path=".env.local")
logger = logging.getLogger("shopping_agent")
logger.setLevel(logging.INFO)


async def entrypoint(ctx: JobContext):
    logger.info(f"connecting to room {ctx.room.name}")
    await ctx.connect(auto_subscribe=AutoSubscribe.SUBSCRIBE_ALL)

    participant = await ctx.wait_for_participant()
    logger.info(f"Started agent for participant {participant.identity}")

    # create conversation id
    conversation_id = str(uuid.uuid4())
    run_multimodal_agent(ctx, participant, conversation_id)

    logger.info("agent started")


def run_multimodal_agent(
    ctx: JobContext, participant: rtc.RemoteParticipant, conversation_id: str
):
    logger.info("Setting up tools")

    # models that can be called in the tools
    images_model = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    web_model = PerplexityChat(
        pplx_api_key=os.environ["PPLX_API_KEY"], pplx_model=PPLX_MODEL
    )
    tool_use_database = AgentDatabase(TOOL_DATABASE_NAME)

    logger.info("starting multimodal agent")

    model = openai.realtime.RealtimeModel(
        model=REALTIME_MODEL,
        instructions=RealTimeModelDriverPrompt.system_message,
        voice=VOICE,
        temperature=REALTIME_TEMPERATURE,
        modalities=["audio", "text"],
        turn_detection=openai.realtime.ServerVadOptions(
            threshold=0.75,
            prefix_padding_ms=300,
            silence_duration_ms=1000,
            create_response=True,
        ),
    )

    tools = AgentTools(
        ctx.room,
        images_model,
        web_model,
        database=tool_use_database,
        user_id=participant.identity,
        conversation_id=conversation_id,
    )

    initial_context = llm.ChatContext().append(
        role="system",
        text=(
            "The name of the person you're speaking to today is {}".format(
                participant.identity
            )
        ),
    )

    agent = MultimodalAgent(
        model=model,
        chat_ctx=initial_context,
        fnc_ctx=tools,
        max_text_response_retries=5,
    )

    # conversation logger
    cp = ConversationLogger(
        model=agent,
        database=tool_use_database,
        user_id=participant.identity,
        conversation_id=conversation_id,
        log=CONVERSATION_LOG_PREFIX + "_{}.txt".format(participant.identity),
    )
    cp.start()
    agent.start(ctx.room, participant)

    session = model.sessions[0]
    session.conversation.item.create(
        llm.ChatMessage(
            role="assistant",
            content="""
            You are speaking to {}
            Say hello addressing the user by name and say that you're here to assist with any shopping needs.
            """.format(participant.identity),
        )
    )
    session.response.create()


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
        )
    )
