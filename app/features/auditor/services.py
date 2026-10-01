from lib.redis import RedisContract

from .entities import AuditReport, classify_key, render_key


class RedisAuditor:
    def __init__(self, redis: RedisContract) -> None:
        self.redis = redis

    async def run(self) -> AuditReport:
        total = 0
        legacy_sessions = 0
        bifrost_cache = 0
        other = []
        async with self.redis.connect() as client:
            async for key in client.scan_iter(count=1000):
                total += 1
                match classify_key(key):
                    case 'legacy_sessions':
                        legacy_sessions += 1
                    case 'bifrost_cache':
                        bifrost_cache += 1
                    case 'other':
                        other.append(render_key(key))
        return AuditReport(total, legacy_sessions, bifrost_cache, tuple(sorted(other)))
