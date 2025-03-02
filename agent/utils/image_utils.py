import io
import base64
from livekit import rtc
from PIL import Image
from livekit.agents import llm, utils


def encode_image(image):
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def convert_base64_to_pil(base64_string):
    try:
        image_bytes = io.BytesIO(base64.b16decode(base64_string))
        image = Image.open(image_bytes)
        return image
    except Exception as e:
        return None


async def get_video_track(room: rtc.Room):
    for participant_id, participant in room.remote_participants.items():
        for track_id, track_publication in participant.track_publications.items():
            if track_publication.track and isinstance(
                track_publication.track, rtc.RemoteVideoTrack
            ):
                return track_publication.track
    raise ValueError("No remote video track found in the room")


async def get_latest_image(room: rtc.Room):
    video_stream = None
    try:
        video_track = await get_video_track(room)
        video_stream = rtc.VideoStream(video_track)
        async for event in video_stream:
            return event.frame
    finally:
        if video_stream:
            await video_stream.aclose()


async def capture_image_from_video_stream(room: rtc.Room):
    latest_image = await get_latest_image(room)

    image_options = utils.images.EncodeOptions()
    image_options.resize_options = utils.images.ResizeOptions(
        width=512, height=512, strategy="scale_aspect_fit"
    )

    res = {"pil_image": None, "b64_image": None}
    if latest_image:
        image_content = llm.ChatImage(image=latest_image)
        encoded_data = base64.b64encode(
            utils.images.encode(image_content.image, image_options)
        )
        pil_image = convert_base64_to_pil(encoded_data)
        res["pil_image"] = pil_image
        res["b64_image"] = encoded_data.decode("utf-8")

    return res
