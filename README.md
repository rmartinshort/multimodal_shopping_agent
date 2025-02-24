
# Multimodal Voice Agent for Online Shopping

This is adapted from one of LiveKit's quickstarts. The agent here is designed to help with your online shopping experience.
It can do the following:

- Search the web (using Perplexity's SONAR API) to get the latest product details
- Open useful links in your browser 
- Take screenshots to give advice about products you're currently looking at
- Capture images from your video stream in case you want to learn more about a product you have at home
- Log all the conversations to a TinyDB, which it also has access to in order to "remember" what you asked about previously

See the LiveKit docs for more details  
[Agents Framework](https://github.com/livekit/agents).
<p>
  <a href="https://cloud.livekit.io/projects/p_/sandbox"><strong>Deploy a sandbox app</strong></a>
  •
  <a href="https://docs.livekit.io/agents/overview/">LiveKit Agents Docs</a>
  •
  <a href="https://livekit.io/cloud">LiveKit Cloud</a>
  •
  <a href="https://blog.livekit.io/">Blog</a>
</p>


## Dev Setup

Clone the repository and install dependencies to a virtual environment:

```console
cd multimodal_shopping_agent
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Set up the environment by making a file called .env.local and adding the following information 

- `LIVEKIT_URL`
- `LIVEKIT_API_KEY`
- `LIVEKIT_API_SECRET`
- `OPENAI_API_KEY`
- `PPLX_API_KEY`

Run the agent:

```console
python3 agent.py dev
```

This agent requires a frontend application to communicate with. To get started, make a meeting room sandbox
so that the agent can have access to audio and video.
