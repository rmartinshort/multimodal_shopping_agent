import asyncio
from dataclasses import dataclass
from datetime import datetime
from typing import Union
from agent.tools.AgentDatabase import AgentDatabase
import aiofiles
from livekit.agents import (
    multimodal,
    utils,
)
from livekit.agents.llm import ChatMessage
from livekit.agents.multimodal.multimodal_agent import EventTypes

"""
Ad adapted from https://github.com/livekit/agents/blob/main/examples/conversation_persistor.py
"""


@dataclass
class EventLog:
    eventname: str | None
    """name of recorded event"""
    time: str = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    """time the event is recorded"""


@dataclass
class TranscriptionLog:
    role: str | None
    """role of the speaker"""
    transcription: str | None
    """transcription of speech"""
    time: str = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    """time the event is recorded"""


class ConversationLogger(utils.EventEmitter[EventTypes]):
    def __init__(
        self,
        *,
        model: multimodal.MultimodalAgent | None,
        log: str | None,
        database: AgentDatabase,
        conversation_id: str,
        user_id: str,
        transcriptions_only: bool = False,
    ):
        """
        Initializes a ConversationLogger instance which records the events and transcriptions of a MultimodalAgent.

        Args:
            model (multimodal.MultimodalAgent): an instance of a MultiModalAgent
            log (str): name of the external file to record events in
            transcriptions_only (bool): a boolean variable to determine if only transcriptions will be recorded, False by default
            user_transcriptions (arr): list of user transcriptions
            agent_transcriptions (arr): list of agent transcriptions
            events (arr): list of all events
            log_q (asyncio.Queue): a queue of EventLog and TranscriptionLog

        """
        super().__init__()

        self._model = model
        self._log = log
        self._transcriptions_only = transcriptions_only

        self._user_transcriptions = []
        self._agent_transcriptions = []
        self._events = []
        self._db = database
        self._user_id = user_id
        self._conversation_id = conversation_id

        self._log_q = asyncio.Queue[Union[EventLog, TranscriptionLog, None]]()

    @property
    def log(self) -> str | None:
        return self._log

    @property
    def model(self) -> multimodal.MultimodalAgent | None:
        return self._model

    @property
    def user_transcriptions(self) -> dict:
        return self._user_transcriptions

    @property
    def agent_transcriptions(self) -> dict:
        return self._agent_transcriptions

    @property
    def events(self) -> dict:
        return self._events

    @property
    def db(self) -> AgentDatabase:
        return self._db

    @property
    def user_id(self) -> str:
        return self._user_id

    @property
    def conversation_id(self) -> str:
        return self._conversation_id

    @log.setter
    def log(self, newlog: str | None) -> None:
        self._log = newlog

    async def _main_atask(self) -> None:
        # Writes to file asynchronously
        while True:
            log = await self._log_q.get()

            if log is None:
                break

            async with aiofiles.open(self._log, "a") as file:
                if type(log) is EventLog and not self._transcriptions_only:
                    self._events.append(log)
                    await file.write("\n" + log.time + " " + log.eventname)

                if type(log) is TranscriptionLog:
                    if log.role == "user":
                        self._user_transcriptions.append(log)
                    else:
                        self._agent_transcriptions.append(log)

                    await file.write(
                        "\n" + log.time + " " + log.role + " " + log.transcription
                    )

    async def aclose(self) -> None:
        # Exits
        self._log_q.put_nowait(None)
        await self._main_task

    def start(self) -> None:
        # Listens for emitted MultimodalAgent events
        self._main_task = asyncio.create_task(self._main_atask())

        @self._model.on("user_started_speaking")
        def _user_started_speaking():
            event = EventLog(eventname="user_started_speaking")
            self._log_q.put_nowait(event)

        @self._model.on("user_stopped_speaking")
        def _user_stopped_speaking():
            event = EventLog(eventname="user_stopped_speaking")
            self._log_q.put_nowait(event)

        @self._model.on("agent_started_speaking")
        def _agent_started_speaking():
            event = EventLog(eventname="agent_started_speaking")
            self._log_q.put_nowait(event)

        @self._model.on("agent_stopped_speaking")
        def _agent_stopped_speaking():
            transcription = TranscriptionLog(
                role="agent",
                transcription=(self._model._playing_handle._tr_fwd.played_text),
            )
            self.db.store_text(
                user_id=self.user_id,
                conversation_id=self.conversation_id,
                tool_id=None,
                data_type="agent_output_text",
                text_data=self._model._playing_handle._tr_fwd.played_text,
            )
            self._log_q.put_nowait(transcription)

            event = EventLog(eventname="agent_stopped_speaking")
            self._log_q.put_nowait(event)

        @self._model.on("user_speech_committed")
        def _user_speech_committed(user_msg: ChatMessage):
            transcription = TranscriptionLog(role="user", transcription=user_msg)
            self.db.store_text(
                user_id=self.user_id,
                conversation_id=self.conversation_id,
                tool_id=None,
                data_type="user_input_text",
                text_data=transcription.transcription,
            )
            self._log_q.put_nowait(transcription)

            event = EventLog(eventname="user_speech_committed")
            self._log_q.put_nowait(event)

        @self._model.on("agent_speech_committed")
        def _agent_speech_committed():
            event = EventLog(eventname="agent_speech_committed")
            self._log_q.put_nowait(event)

        @self._model.on("agent_speech_interrupted")
        def _agent_speech_interrupted():
            event = EventLog(eventname="agent_speech_interrupted")
            self._log_q.put_nowait(event)

        @self._model.on("function_calls_collected")
        def _function_calls_collected():
            event = EventLog(eventname="function_calls_collected")
            self._log_q.put_nowait(event)

        @self._model.on("function_calls_finished")
        def _function_calls_finished():
            event = EventLog(eventname="function_calls_finished")
            self._log_q.put_nowait(event)
