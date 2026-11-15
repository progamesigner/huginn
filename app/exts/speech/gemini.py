import io
import wave
from email.message import Message

from pydantic import SecretStr

from app.exts.gemini.clients import GeminiClient
from app.exts.gemini.exceptions import GeminiGenerationError
from lib.http import HTTPClientContract
from lib.media.encoding import encode_opus

from .exceptions import SpeechSynthesisError

_MAX_BYTES = 20 * 1024 * 1024


class GeminiSpeechClient:
    def __init__(
        self,
        http: HTTPClientContract,
        api_key: SecretStr,
        base_url: str,
        model: str,
    ) -> None:
        self.client = GeminiClient(http, api_key, base_url, model, _MAX_BYTES)

    async def synthesize(self, text: str, voice: str, response_format: str) -> bytes:
        if response_format != 'opus':
            raise SpeechSynthesisError('Gemini speech supports only opus output')
        try:
            result = await self.client.generate(
                text,
                {
                    'responseModalities': ['AUDIO'],
                    'speechConfig': {
                        'voiceConfig': {'prebuiltVoiceConfig': {'voiceName': voice}},
                    },
                },
                'audio/',
            )
            data, rate = _pcm(result.data, result.mime)
            return await encode_opus(data, sample_rate=rate)
        except (GeminiGenerationError, ValueError, wave.Error, EOFError) as error:
            raise SpeechSynthesisError(str(error)) from error


def _pcm(data: bytes, mime: str) -> tuple[bytes, int]:
    header = Message()
    header['content-type'] = mime
    media_type = header.get_content_type().lower()
    if media_type in ('audio/wav', 'audio/x-wav'):
        with wave.open(io.BytesIO(data), 'rb') as audio:
            if audio.getnchannels() != 1 or audio.getsampwidth() != 2:
                raise ValueError('Gemini speech must contain mono 16-bit PCM')
            rate = audio.getframerate()
            pcm = audio.readframes(audio.getnframes())
    elif media_type == 'audio/l16':
        parameter = header.get_param('rate') or '24000'
        if not isinstance(parameter, str):
            raise ValueError('Gemini returned an invalid sample rate')
        rate = int(parameter)
        if header.get_param('codec') not in (None, 'pcm'):
            raise ValueError('Gemini returned an unsupported audio codec')
        pcm = data
    else:
        raise ValueError('Gemini returned an unsupported audio format')
    if rate not in (8000, 16000, 24000, 48000) or not pcm or len(pcm) % 2:
        raise ValueError('Gemini returned invalid mono 16-bit PCM')
    return pcm, rate
