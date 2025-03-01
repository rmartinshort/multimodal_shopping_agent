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
from agent.tools.PerplexityChat import PerplexityChat
from agent.tools.AgentTools import AgentTools
from agent.prompts import RealTimeModelDriverPrompt
from agent.tools.AgentDatabase import AgentDatabase
from agent.tools.AgentConversationLogger import ConversationLogger
from agent.config import (
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
    """
    Sets up and runs a multimodal agent for a given participant in a LiveKit room.

    This function initializes the necessary tools, models, and database connections
    for the agent. It then configures the agent with specific instructions, voice,
    and temperature settings. The agent is equipped with tools for image generation,
    web search, and database interaction, enabling it to assist the participant
    with various tasks.

    Args:
         ctx: The JobContext object providing access to the LiveKit room.
         participant: The rtc.RemoteParticipant object representing the user in the room.
         conversation_id: A unique identifier for the conversation.

    Returns:
         None. This function starts the agent and logs the conversation.

    Raises:
         KeyError: If any required environment variables (e.g., OPENAI_API_KEY, PPLX_API_KEY) are not set.

    Example:
         ```
         # Assuming 'ctx' is a JobContext, 'participant' is an rtc.RemoteParticipant,
         # and 'conversation_id' is a string.
         run_multimodal_agent(ctx, participant, "unique_conversation_id")
         ```
    """
    logger.info("Setting up tools")

    # models that can be called in the tools
    images_model = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    web_model = PerplexityChat(
        pplx_api_key=os.environ["PPLX_API_KEY"], pplx_model=PPLX_MODEL
    )
    # set up database
    conversation_and_tool_use_database = AgentDatabase(TOOL_DATABASE_NAME)

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
        database=conversation_and_tool_use_database,
        user_id=participant.identity,
        conversation_id=conversation_id,
    )

    initial_context = llm.ChatContext().append(
        role="system",
        text=(
            "Do not hallucinate. If you don't understand what the user just said then ask them to repeat it"
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
        database=conversation_and_tool_use_database,
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
            Here is the name of the person you're speaking to: {}
            Say hello addressing the user by name and say that you're here to assist with any shopping needs.
            Then explain briefly the things that you can do. 
            
            If you decide to call a tool, immediately before issuing the call you should say briefly what you're about to 
            do and mention that it may take a few seconds.
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
